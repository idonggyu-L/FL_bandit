#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Federated learning runner with rule-based embedding filtering + bandit rules.

This file intentionally does not modify ``main_fed.py`` or the existing defense
modules. It mirrors the original training loop and replaces only the aggregation
step with:

    local client model -> validation embeddings -> rule-based prefilter
    -> contextual bandit chooses a selection rule -> FedAvg over selected client models

The validation embeddings are computed by replaying each uploaded local model on
the same small server validation set used by FLTrust/FLARE in this repository.
That is the simulator equivalent of each client producing embeddings for an
agreed validation set and sending only the embeddings/updates to the server.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from main_fed import central_dataset_iid, test_mkdir, write_file
from models.Attacker import attacker
from models.Fed import FedAvg
from models.Nets import ResNet18, get_model, vgg11, vgg19_bn
from models.Update import DatasetSplit, LocalUpdate
from models.resnet20 import resnet20
from models.test import test_img
from utils.defense import get_update, multi_krum
from utils.info import get_base_info, print_exp_details
from utils.options import args_parser
from utils.rl_module import (
    BanditConfig,
    RuleAction,
    build_rule_actions,
    create_rule_bandit_agent,
)
from utils.sampling import cifar_noniid, mnist_iid, mnist_noniid

EPS = 1e-12
RULE_RL_STATE_DIM = 14
RULE_RL_ROLLOUTS_PER_ROUND = 50


def to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def init_wandb(args: Any, base_info: str, validation_size: int, eval_size: int):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb was requested with --wandb, but wandb is not installed") from exc

    config = {}
    for key, value in vars(args).items():
        if isinstance(value, torch.device):
            value = str(value)
        config[key] = value
    config.update(
        {
            "validation_size": int(validation_size),
            "eval_size": int(eval_size),
            "rule_rl_state_dim": RULE_RL_STATE_DIM,
        }
    )
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or base_info,
        mode=args.wandb_mode,
        config=config,
    )


def build_wandb_metrics(
    iter_idx: int,
    loss_avg: Any,
    acc_test: Any,
    back_acc: Any,
    rule_rl_meta: Dict[str, Any],
    round_attack_count: int,
) -> Dict[str, Any]:
    action = rule_rl_meta["action"]
    prefiltered = [int(i) for i in rule_rl_meta["prefiltered_clients"]]
    selected = [int(i) for i in rule_rl_meta["selected_clients"]]
    num_clients = int(rule_rl_meta["num_clients"])
    round_attack_count = int(round_attack_count)
    benign_count = max(num_clients - round_attack_count, 0)

    prefilter_malicious = sum(1 for idx in prefiltered if idx < round_attack_count)
    final_malicious = sum(1 for idx in selected if idx < round_attack_count)
    prefilter_benign = len(prefiltered) - prefilter_malicious
    final_benign = len(selected) - final_malicious

    score_type_id = {"l2_to_center": 0, "cosine_to_center": 1, "knn_distance": 2}
    center_type_id = {"mean": 0, "median": 1, "medoid": 2, "none": 3}

    metrics: Dict[str, Any] = {
        "round": int(iter_idx),
        "train/loss_avg": to_float(loss_avg),
        "eval/main_accuracy": to_float(acc_test),
        "eval/backdoor_accuracy": to_float(back_acc),
        "attack/active": float(round_attack_count > 0),
        "attack/round_malicious_clients": round_attack_count,
        "bandit/validation_accuracy": float(rule_rl_meta["validation_accuracy"]),
        "bandit/baseline_validation_accuracy": float(rule_rl_meta["baseline_validation_accuracy"]),
        "bandit/separation": float(rule_rl_meta["separation"]),
        "bandit/baseline_separation": float(rule_rl_meta["baseline_separation"]),
        "bandit/acc_delta": float(rule_rl_meta["acc_delta"]),
        "bandit/sep_delta": float(rule_rl_meta["sep_delta"]),
        "bandit/proto_weight": float(rule_rl_meta["proto_weight"]),
        "bandit/reward": float(rule_rl_meta["reward"]),
        "bandit/rollout_reward_mean": float(rule_rl_meta["rollout_reward_mean"]),
        "bandit/rollout_reward_max": float(rule_rl_meta["rollout_reward_max"]),
        "bandit/num_rollouts_executed": int(rule_rl_meta["num_rollouts_executed"]),
        "bandit/num_transitions_added": int(rule_rl_meta["num_transitions_added"]),
        "bandit/action_id": int(rule_rl_meta["action_id"]),
        "bandit/action_score_type_id": score_type_id.get(action["score_type"], -1),
        "bandit/action_center_type_id": center_type_id.get(action["center_type"], -1),
        "bandit/action_drop_inner_ratio": float(action["drop_inner_ratio"]),
        "bandit/action_keep_outer_ratio": float(action["keep_outer_ratio"]),
        "bandit/action_knn_k": int(action["knn_k"]),
        "bandit/exploration": float(rule_rl_meta["exploration"]),
        "bandit/epsilon": float(rule_rl_meta["epsilon"]),
        "bandit/prefiltered_count": len(prefiltered),
        "bandit/selected_count": len(selected),
        "bandit/prefiltered_fraction": len(prefiltered) / max(num_clients, 1),
        "bandit/selected_fraction": len(selected) / max(num_clients, 1),
        "defense/prefilter_malicious_survival_count": prefilter_malicious,
        "defense/final_malicious_survival_count": final_malicious,
        "defense/prefilter_benign_survival_count": prefilter_benign,
        "defense/final_benign_selected_count": final_benign,
        "defense/prefilter_malicious_survival_ratio": (
            prefilter_malicious / round_attack_count if round_attack_count else 0.0
        ),
        "defense/final_malicious_survival_ratio": (
            final_malicious / round_attack_count if round_attack_count else 0.0
        ),
        "defense/prefilter_benign_survival_ratio": (
            prefilter_benign / benign_count if benign_count else 0.0
        ),
        "defense/final_benign_selected_ratio": (
            final_benign / benign_count if benign_count else 0.0
        ),
        "defense/final_malicious_selected_fraction": (
            final_malicious / len(selected) if selected else 0.0
        ),
    }

    metrics["bandit/action_score_type"] = action["score_type"]
    metrics["bandit/action_center_type"] = action["center_type"]
    metrics["bandit/bandit_mode"] = rule_rl_meta["bandit_mode"]

    train_metrics = rule_rl_meta.get("train_metrics") or {}
    for key, value in train_metrics.items():
        metrics[f"bandit/train_{key}"] = to_float(value)

    # trigger-free class-confidence diagnostics; compare against attack/active
    for key, value in (rule_rl_meta.get("diag") or {}).items():
        metrics[f"diag/{key}"] = to_float(value)
    return metrics


@dataclass
class RuleFilterConfig:
    """Conservative multi-Krum prefilter with a min-keep floor."""

    min_keep_ratio: float = 0.5
    min_keep_clients: int = 2


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale < EPS:
        scale = float(np.std(values)) + EPS
    return np.abs(values - median) / scale


def pairwise_l2(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    diff = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diff, axis=2)


def cosine_distance_to_center(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    x_norm = np.linalg.norm(x, axis=1)
    center_norm = float(np.linalg.norm(center))
    denom = np.maximum(x_norm * center_norm, EPS)
    cos = np.clip((x @ center) / denom, -1.0, 1.0)
    return 1.0 - cos


def knn_distance_scores(x: np.ndarray, k: int) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    dists = pairwise_l2(x)
    np.fill_diagonal(dists, np.inf)
    k = max(1, min(k, n - 1))
    nearest = np.partition(dists, kth=k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1)


def center_from_embeddings(x: np.ndarray, center_type: str) -> np.ndarray:
    if center_type == "mean":
        return x.mean(axis=0)
    if center_type == "median":
        return np.median(x, axis=0)
    if center_type == "medoid":
        dists = pairwise_l2(x)
        return x[int(np.argmin(dists.mean(axis=1)))]
    raise ValueError(f"Unsupported center_type: {center_type}")


def score_embeddings_by_action(x: np.ndarray, action: RuleAction) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=np.float64)
    if action.score_type == "knn_distance":
        return knn_distance_scores(x, action.knn_k)

    center = center_from_embeddings(x, action.center_type)
    if action.score_type == "l2_to_center":
        return np.linalg.norm(x - center, axis=1)
    if action.score_type == "cosine_to_center":
        return cosine_distance_to_center(x, center)
    raise ValueError(f"Unsupported score_type: {action.score_type}")


def select_by_rule_action(x: np.ndarray, action: RuleAction) -> List[int]:
    n = len(x)
    if n == 0:
        return []
    if n == 1:
        return [0]

    scores = score_embeddings_by_action(x, action)
    order = np.argsort(scores)
    start = int(np.floor(n * action.drop_inner_ratio))
    end = int(np.ceil(n * action.keep_outer_ratio))
    start = min(max(start, 0), n - 1)
    end = min(max(end, start + 1), n)
    return [int(i) for i in order[start:end]]


def prototype_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    """Prototype-based class-separation score in [-1, 1].

    For each validation sample: ``a`` = L2 distance to its own class prototype
    (the mean feature of its class), ``b`` = L2 distance to the nearest *other*
    class prototype. The score is the mean over samples of ``(b - a)/max(a, b)``
    -- a simplified silhouette that measures prototype consistency: high when
    every sample sits close to its own class prototype and far from all others.

    This is the stage-2 reward's "prototype consistency" term. A backdoor that
    warps the feature space (e.g. collapsing one class toward an attractor, or
    pulling triggered inputs across a boundary) degrades this even when overall
    validation accuracy still looks healthy, so it complements the accuracy term.

    Needs at least two populated classes; returns 0.0 otherwise (no signal).
    """
    if features.shape[0] == 0:
        return 0.0
    classes = np.unique(labels)
    if classes.size < 2:
        return 0.0
    protos = np.stack([features[labels == c].mean(axis=0) for c in classes], axis=0)
    cls_index = {int(c): i for i, c in enumerate(classes)}
    diff = features[:, None, :] - protos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)  # (n_samples, n_classes)
    own = np.array([cls_index[int(c)] for c in labels])
    rows = np.arange(len(labels))
    a = dist[rows, own]
    other = dist.copy()
    other[rows, own] = np.inf
    b = other.min(axis=1)
    denom = np.maximum(a, b)
    sil = np.where(denom > EPS, (b - a) / denom, 0.0)
    return float(np.mean(sil))


def summarize_feature_tensor(features: torch.Tensor) -> np.ndarray:
    flat = features.detach().float().cpu().view(features.size(0), -1)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False)
    return torch.cat([mean, std], dim=0).numpy().astype(np.float64)


def update_l2_norm(update: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, value in update.items():
        suffix = key.split(".")[-1]
        if suffix in {"num_batches_tracked", "running_mean", "running_var"}:
            continue
        if not torch.is_floating_point(value):
            continue
        total += float(torch.sum(value.detach().float().cpu() ** 2).item())
    return float(math.sqrt(max(total, 0.0)))


def build_rule_state(
    embeddings: np.ndarray,
    prefiltered_indices: Sequence[int],
    update_norms: Sequence[float],
) -> np.ndarray:
    n = len(embeddings)
    if n == 0:
        return np.zeros(RULE_RL_STATE_DIM, dtype=np.float32)

    x = np.asarray(embeddings, dtype=np.float64)
    center = np.median(x, axis=0)
    l2 = np.linalg.norm(x - center, axis=1)
    l2_scale = float(np.median(l2) + np.median(np.abs(l2 - np.median(l2))) + EPS)
    pref = np.asarray(list(prefiltered_indices), dtype=np.int64)
    if pref.size == 0:
        pref = np.arange(n, dtype=np.int64)

    cos_dist = cosine_distance_to_center(x, center)
    knn_dist = knn_distance_scores(x, k=3)
    norms = np.asarray(update_norms, dtype=np.float64)
    if norms.size != n:
        norms = np.zeros(n, dtype=np.float64)

    state = np.array(
        [
            min(n / 100.0, 1.0),
            len(pref) / max(n, 1),
            float(np.mean(l2) / l2_scale),
            float(np.std(l2) / l2_scale),
            float(np.max(l2) / l2_scale),
            float(np.mean(l2[pref]) / l2_scale),
            float(np.std(l2[pref]) / l2_scale),
            float(np.max(l2[pref]) / l2_scale),
            float(np.mean(cos_dist)),
            float(np.std(cos_dist)),
            float(np.mean(knn_dist) / l2_scale),
            float(np.std(knn_dist) / l2_scale),
            float(np.log1p(np.mean(norms))),
            float(np.std(norms) / (np.mean(norms) + EPS)),
        ],
        dtype=np.float32,
    )
    state[~np.isfinite(state)] = 0.0
    return np.clip(state, -10.0, 10.0).astype(np.float32)


class RuleBasedEmbeddingFilter:
    """multi-Krum based prefilter with a conservative min-keep floor.

    Uses the existing rule-based defense (``utils.defense.multi_krum``) as the
    Stage-1 filter. If multi-Krum prunes below ``min_keep_ratio`` of clients,
    the missing slots are backfilled with the clients closest to the embedding
    median so the bandit always has a sufficiently large candidate pool.
    """

    def __init__(self, config: RuleFilterConfig):
        self.config = config

    def select(
        self,
        args: Any,
        local_updates: Sequence[Dict[str, torch.Tensor]],
        embeddings: np.ndarray,
    ) -> List[int]:
        n = len(local_updates)
        if n <= self.config.min_keep_clients:
            return list(range(n))

        num_clients = max(int(args.frac * args.num_users), 1)
        n_attackers = max(int(args.malicious * num_clients), 1)

        if n > 2 * n_attackers + 2:
            kept = [int(i) for i in multi_krum(
                list(local_updates), n_attackers, args, multi_k=True
            )]
        else:
            kept = list(range(n))

        min_keep = max(
            self.config.min_keep_clients,
            int(math.ceil(n * self.config.min_keep_ratio)),
        )
        min_keep = min(min_keep, n)

        if len(kept) < min_keep:
            x = np.asarray(embeddings, dtype=np.float64)
            center = np.median(x, axis=0)
            embedding_dist = np.linalg.norm(x - center, axis=1)
            kept_set = set(kept)
            for i in np.argsort(embedding_dist):
                idx = int(i)
                if idx not in kept_set:
                    kept.append(idx)
                    kept_set.add(idx)
                    if len(kept) >= min_keep:
                        break

        return sorted(set(kept))


class RuleRLServer:
    def __init__(
        self,
        args: Any,
        validation_dataset: Any,
        validation_indices: Sequence[int],
        filter_config: Optional[RuleFilterConfig] = None,
    ):
        self.args = args
        self.validation_dataset = validation_dataset
        self.validation_indices = list(validation_indices)
        self.filter = RuleBasedEmbeddingFilter(filter_config or RuleFilterConfig())

        actions = build_rule_actions(
            ratio_pairs=(
                (0.0, 0.5),
                (0.0, 0.8),
                (0.1, 0.8),
                (0.2, 0.9),
                (0.0, 1.0),
            ),
            score_types=("l2_to_center", "cosine_to_center", "knn_distance"),
            center_types=("mean", "median", "medoid"),
            knn_k=3,
        )
        config = BanditConfig(
            state_dim=RULE_RL_STATE_DIM,
            num_actions=len(actions),
            mode=getattr(args, "rule_bandit_mode", "LinUCB"),
            lr=float(getattr(args, "rule_bandit_lr", 1e-3)),
            batch_size=int(getattr(args, "rule_bandit_batch_size", 16)),
            replay_capacity=5000,
            lambda_reg=float(getattr(args, "rule_bandit_lambda", 1.0)),
            alpha=float(getattr(args, "rule_bandit_alpha", 1.0)),
            nu=float(getattr(args, "rule_bandit_nu", 1.0)),
            hidden_dim=int(getattr(args, "rule_bandit_hidden_dim", 128)),
            feature_dim=int(getattr(args, "rule_bandit_feature_dim", 64)),
            train_steps_per_update=int(getattr(args, "rule_bandit_train_steps", 1)),
            seed=args.seed,
            device=str(args.device),
        )
        self.agent = create_rule_bandit_agent(actions=actions, config=config)
        self.rollouts_per_round = int(
            getattr(
                args,
                "rule_bandit_rollouts_per_round",
                getattr(args, "rule_rl_rollouts_per_round", RULE_RL_ROLLOUTS_PER_ROUND),
            )
        )
        # True-bandit evaluation budget: how many exploratory actions to actually
        # evaluate (and learn from) per round. Keeping this small is the whole
        # point of using a bandit -- we do NOT evaluate every rule each round.
        self.evals_per_round = max(int(getattr(args, "rule_bandit_evals_per_round", 1)), 1)
        # Reward = accuracy_delta + proto_weight * separation_delta. proto_weight=0
        # recovers the pure-accuracy bandit (clean ablation baseline).
        self.proto_weight = float(getattr(args, "rule_bandit_proto_weight", 1.0))
        self.last_validation_accuracy: Optional[float] = None
        self.last_separation: Optional[float] = None
        self.history: List[Dict[str, Any]] = []
        self._scratch_net: Optional[torch.nn.Module] = None
        self._val_loader: Optional[DataLoader] = None
        self._val_batches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    def _get_scratch_net(self, net_template: torch.nn.Module) -> torch.nn.Module:
        if self._scratch_net is None:
            self._scratch_net = copy.deepcopy(net_template).to(self.args.device)
        return self._scratch_net

    def _get_val_batches(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if self._val_batches is None:
            loader = DataLoader(
                DatasetSplit(self.validation_dataset, self.validation_indices),
                batch_size=self.args.bs,
                shuffle=False,
            )
            self._val_batches = [
                (data.to(self.args.device), target.to(self.args.device))
                for data, target in loader
            ]
        return self._val_batches

    @torch.no_grad()
    def collect_client_embeddings(
        self,
        net_template: torch.nn.Module,
        local_weights: Sequence[Dict[str, torch.Tensor]],
    ) -> np.ndarray:
        if not hasattr(net_template, "get_feature"):
            raise AttributeError("rule_rl requires models with a get_feature(images) method")

        batches = self._get_val_batches()
        probe_net = self._get_scratch_net(net_template)
        embeddings: List[np.ndarray] = []

        for local_state in local_weights:
            probe_net.load_state_dict(local_state)
            probe_net.eval()
            features_list = []
            for images, _ in batches:
                features = probe_net.get_feature(images)
                features_list.append(features.detach())
            features_tensor = torch.cat(features_list, dim=0)
            embeddings.append(summarize_feature_tensor(features_tensor))
        return np.stack(embeddings, axis=0)

    @torch.no_grad()
    def validation_accuracy(
        self,
        net: torch.nn.Module,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> float:
        eval_net = self._get_scratch_net(net)
        if state_dict is None:
            state_dict = net.state_dict()
        eval_net.load_state_dict(state_dict)
        eval_net.eval()

        batches = self._get_val_batches()
        correct = 0
        total = 0
        for data, target in batches:
            logits = eval_net(data)
            pred = logits.data.max(1, keepdim=True)[1]
            correct += int(pred.eq(target.data.view_as(pred)).long().cpu().sum().item())
            total += int(target.numel())
        return 100.0 * correct / max(total, 1)

    @torch.no_grad()
    def evaluate_global(
        self,
        net: torch.nn.Module,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[float, float]:
        """Validation accuracy (%) and prototype separation, in one model load.

        Returns ``(accuracy, separation)`` where ``separation`` is the
        prototype-based silhouette (see :func:`prototype_silhouette`). Both are
        read from the same clean validation pass so adding the prototype reward
        term costs one extra ``get_feature`` per batch, not a second eval loop.
        """
        eval_net = self._get_scratch_net(net)
        if state_dict is None:
            state_dict = net.state_dict()
        eval_net.load_state_dict(state_dict)
        eval_net.eval()

        batches = self._get_val_batches()
        correct = 0
        total = 0
        feats: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for data, target in batches:
            logits = eval_net(data)
            pred = logits.data.max(1, keepdim=True)[1]
            correct += int(pred.eq(target.data.view_as(pred)).long().cpu().sum().item())
            total += int(target.numel())
            features = eval_net.get_feature(data)
            feats.append(features.detach().float().cpu().view(features.size(0), -1))
            labels.append(target.detach().cpu())
        accuracy = 100.0 * correct / max(total, 1)
        if not feats:
            return accuracy, 0.0
        features_np = torch.cat(feats, dim=0).numpy().astype(np.float64)
        labels_np = torch.cat(labels, dim=0).numpy().astype(np.int64)
        separation = prototype_silhouette(features_np, labels_np)
        return accuracy, separation

    @torch.no_grad()
    def class_confidence_stats(
        self,
        net: torch.nn.Module,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Trigger-free per-class statistics on the clean validation set.

        Diagnostic only -- nothing here feeds the reward yet. The point is to
        check whether any of these quantities separates attack rounds from clean
        rounds before we commit one as a reward term. The intuition under test:
        a backdoor that makes one class an over-confident attractor should leave
        that class as an outlier in mean confidence / low entropy even on clean
        inputs.
        """
        eval_net = self._get_scratch_net(net)
        if state_dict is None:
            state_dict = net.state_dict()
        eval_net.load_state_dict(state_dict)
        eval_net.eval()

        batches = self._get_val_batches()
        num_classes: Optional[int] = None
        conf_sum = ent_sum = correct = count = None
        for data, target in batches:
            probs = torch.softmax(eval_net(data), dim=1)
            if num_classes is None:
                num_classes = probs.size(1)
                conf_sum = np.zeros(num_classes)
                ent_sum = np.zeros(num_classes)
                correct = np.zeros(num_classes)
                count = np.zeros(num_classes)
            conf, pred = probs.max(dim=1)
            entropy = -(probs * torch.log(probs + EPS)).sum(dim=1)
            t = target.cpu().numpy()
            c = conf.cpu().numpy()
            e = entropy.cpu().numpy()
            hit = pred.eq(target).cpu().numpy().astype(np.float64)
            for cls in range(num_classes):
                mask = t == cls
                if not np.any(mask):
                    continue
                conf_sum[cls] += c[mask].sum()
                ent_sum[cls] += e[mask].sum()
                correct[cls] += hit[mask].sum()
                count[cls] += int(mask.sum())

        if num_classes is None or not np.any(count > 0):
            return {"conf_var": 0.0, "conf_max_z": 0.0, "acc_var": 0.0,
                    "ent_mean": 0.0, "ent_min": 0.0}

        valid = count > 0
        per_conf = (conf_sum / np.maximum(count, 1))[valid]
        per_ent = (ent_sum / np.maximum(count, 1))[valid]
        per_acc = (correct / np.maximum(count, 1))[valid]
        conf_mean = float(np.mean(per_conf))
        conf_std = float(np.std(per_conf))
        return {
            "conf_var": float(np.var(per_conf)),
            # how much the most extreme class deviates from the mean (in stds):
            # a single over-confident attractor class shows up as a large z.
            "conf_max_z": float(np.max(np.abs(per_conf - conf_mean)) / (conf_std + EPS)),
            "acc_var": float(np.var(per_acc)),
            "ent_mean": float(np.mean(per_ent)),
            "ent_min": float(np.min(per_ent)),
        }

    @torch.no_grad()
    def per_client_stats(
        self,
        net_template: torch.nn.Module,
        local_weights: Sequence[Dict[str, torch.Tensor]],
        embeddings: np.ndarray,
        n_malicious: int,
    ) -> List[Dict[str, Any]]:
        """Per-client outlierness on clean data -- the granularity stage-2 uses.

        For each *individual* client we record the rule-family scores the bandit
        actually ranks on (``rule_*``: l2/cosine/knn to mean/median/medoid
        centers in embedding space) plus the client's own clean class-confidence
        stats. This faithfully tests whether *any* rule in the current family can
        separate malicious from benign -- not just l2-to-center. Each record
        carries the ground-truth malicious flag; malicious vs benign are compared
        within the same round, so it is confound- and dilution-free.

        Note: the action space is a rule *selector*; rules need not be distance
        based. If none of these geometric scores separate, that argues for
        extending the rule family (or a different signal space), not just tuning.
        """
        x = np.asarray(embeddings, dtype=np.float64)
        rule_scores: Dict[str, np.ndarray] = {}
        for center_type in ("mean", "median", "medoid"):
            center = center_from_embeddings(x, center_type)
            rule_scores[f"l2_{center_type}"] = np.linalg.norm(x - center, axis=1)
            rule_scores[f"cos_{center_type}"] = cosine_distance_to_center(x, center)
        rule_scores["knn"] = knn_distance_scores(x, k=3)

        records: List[Dict[str, Any]] = []
        for i, w in enumerate(local_weights):
            stats = self.class_confidence_stats(net_template, w)
            rec: Dict[str, Any] = {"malicious": bool(i < n_malicious)}
            for name, score in rule_scores.items():
                rec[f"rule_{name}"] = float(score[i])
            rec.update({k: float(v) for k, v in stats.items()})
            records.append(rec)
        return records

    def select_clients_for_action(
        self,
        embeddings: np.ndarray,
        prefiltered: Sequence[int],
        action: RuleAction,
    ) -> List[int]:
        if len(prefiltered) == 0:
            return list(range(len(embeddings)))

        prefiltered = list(prefiltered)
        candidate_embeddings = embeddings[prefiltered]
        selected_relative = select_by_rule_action(candidate_embeddings, action)
        selected_clients = [prefiltered[i] for i in selected_relative]
        if len(selected_clients) == 0:
            selected_clients = prefiltered
        return [int(i) for i in selected_clients]

    def aggregate(
        self,
        round_idx: int,
        net_template: torch.nn.Module,
        local_weights: Sequence[Dict[str, torch.Tensor]],
        local_updates: Sequence[Dict[str, torch.Tensor]],
        n_malicious: int = 0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if len(local_weights) == 0:
            raise ValueError("No local weights were provided for aggregation")

        embeddings = self.collect_client_embeddings(net_template, local_weights)
        update_norms = [update_l2_norm(update) for update in local_updates]
        prefiltered = self.filter.select(self.args, local_updates, embeddings)

        state = build_rule_state(embeddings, prefiltered, update_norms)

        if self.last_validation_accuracy is None:
            base_acc, base_sep = self.evaluate_global(net_template)
            self.last_validation_accuracy = base_acc
            self.last_separation = base_sep
        baseline_accuracy = float(self.last_validation_accuracy)
        baseline_separation = float(self.last_separation)

        train_metrics: Optional[Dict[str, float]] = None
        evaluated: Dict[int, Dict[str, Any]] = {}
        eval_records: List[Dict[str, Any]] = []

        def evaluate_action(action_id: int) -> Dict[str, Any]:
            """Evaluate a rule once on the validation set and learn from it.

            Each distinct action is only ever evaluated (and pushed as a
            transition) once per round; repeated picks reuse the cached result
            so the bandit never spends extra validation passes.
            """
            nonlocal train_metrics
            if action_id in evaluated:
                return evaluated[action_id]
            action = self.agent.get_action_object(action_id)
            selected_clients = self.select_clients_for_action(
                embeddings, prefiltered, action
            )
            candidate_global = FedAvg([local_weights[i] for i in selected_clients])
            candidate_accuracy, candidate_separation = self.evaluate_global(
                net_template, candidate_global
            )
            acc_delta = (candidate_accuracy - baseline_accuracy) / 100.0
            sep_delta = candidate_separation - baseline_separation
            reward = float(acc_delta + self.proto_weight * sep_delta)
            record = {
                "action_id": int(action_id),
                "action": action,
                "selected_clients": [int(i) for i in selected_clients],
                "global": candidate_global,
                "validation_accuracy": float(candidate_accuracy),
                "separation": float(candidate_separation),
                "acc_delta": float(acc_delta),
                "sep_delta": float(sep_delta),
                "reward": reward,
            }
            evaluated[action_id] = record
            eval_records.append(record)

            self.agent.push_transition(state, action_id, reward, state, True)
            updated_metrics = self.agent.update()
            if updated_metrics is not None:
                train_metrics = updated_metrics
            return record

        # --- Exploration: spend a small evaluation budget so the bandit can
        #     learn which rule fits this context. With evals_per_round=1 this is
        #     a single exploratory validation pass per round. ---
        for _ in range(self.evals_per_round):
            explore_action_id = self.agent.select_action(state)  # exploration on
            evaluate_action(explore_action_id)

        # --- Commit: deploy the bandit's greedy best-estimate action. This
        #     protects the global model from a single bad exploratory pick. It is
        #     evaluated only if the exploration budget did not already cover it. ---
        final_action_id = self.agent.select_action(state, eval_mode=True)
        final_record = evaluate_action(final_action_id)
        final_action = final_record["action"]
        final_selected_clients = final_record["selected_clients"]
        new_global = final_record["global"]
        final_accuracy = final_record["validation_accuracy"]
        final_separation = final_record["separation"]
        final_reward = final_record["reward"]
        self.last_validation_accuracy = final_accuracy
        self.last_separation = final_separation

        if eval_records:
            best_record = max(
                eval_records, key=lambda item: item["validation_accuracy"]
            )
            best_rollout = {
                "action_id": int(best_record["action_id"]),
                "selected_clients": list(best_record["selected_clients"]),
                "reward": float(best_record["reward"]),
                "validation_accuracy": float(best_record["validation_accuracy"]),
            }
            rollout_rewards = [item["reward"] for item in eval_records]
        else:
            best_rollout = None
            rollout_rewards = []

        # --- Diagnostic: trigger-free class-confidence signal. Measured on the
        #     unfiltered full-pool aggregation (max malicious influence) and on
        #     the committed model (after filtering). Correlating these against
        #     the round's attack ground-truth tells us whether the signal is
        #     worth turning into a reward term. Not fed to the bandit. ---
        diag: Dict[str, float] = {}
        per_client: List[Dict[str, Any]] = []
        if getattr(self.args, "rule_bandit_diag", True):
            full_pool_global = FedAvg(list(local_weights))
            full_stats = self.class_confidence_stats(net_template, full_pool_global)
            committed_stats = self.class_confidence_stats(net_template, new_global)
            for key, value in full_stats.items():
                diag[f"full_{key}"] = float(value)
            for key, value in committed_stats.items():
                diag[f"committed_{key}"] = float(value)
                diag[f"delta_{key}"] = float(full_stats[key] - value)

            # Oracle within-round counterfactual (validation only; uses the
            # ground-truth that the first n_malicious uploads are the attackers).
            # Same round => same model maturity, so the gap is purely the effect
            # of including malicious clients. This is the confound-free Q1 test.
            if 0 < n_malicious < len(local_weights):
                benign_global = FedAvg(list(local_weights)[n_malicious:])
                benign_stats = self.class_confidence_stats(net_template, benign_global)
                for key, value in full_stats.items():
                    diag[f"oraclegap_{key}"] = float(value - benign_stats[key])

                # per-client (stage-2 granularity): is each malicious client an
                # outlier vs benign in embedding distance / clean confidence?
                per_client = self.per_client_stats(
                    net_template, local_weights, embeddings, n_malicious
                )

        meta = {
            "round": int(round_idx),
            "diag": diag,
            "per_client": per_client,
            "bandit_mode": self.agent.mode,
            "num_clients": int(len(local_weights)),
            "prefiltered_clients": [int(i) for i in prefiltered],
            "selected_clients": [int(i) for i in final_selected_clients],
            "action_id": int(final_action_id),
            "action": {
                "score_type": final_action.score_type,
                "center_type": final_action.center_type,
                "drop_inner_ratio": float(final_action.drop_inner_ratio),
                "keep_outer_ratio": float(final_action.keep_outer_ratio),
                "knn_k": int(final_action.knn_k),
            },
            "reward": final_reward,
            "validation_accuracy": float(final_accuracy),
            "baseline_validation_accuracy": baseline_accuracy,
            "separation": float(final_separation),
            "baseline_separation": baseline_separation,
            "acc_delta": float(final_record["acc_delta"]),
            "sep_delta": float(final_record["sep_delta"]),
            "proto_weight": float(self.proto_weight),
            "rollouts_per_round": int(self.evals_per_round),
            "num_rollouts_executed": int(len(eval_records)),
            "num_transitions_added": int(len(evaluated)),
            "rollout_reward_mean": float(np.mean(rollout_rewards))
            if rollout_rewards
            else 0.0,
            "rollout_reward_max": float(np.max(rollout_rewards))
            if rollout_rewards
            else 0.0,
            "best_rollout": best_rollout,
            "rollouts": [
                {
                    "action_id": int(r["action_id"]),
                    "selected_clients": list(r["selected_clients"]),
                    "reward": float(r["reward"]),
                    "validation_accuracy": float(r["validation_accuracy"]),
                }
                for r in eval_records
            ],
            "epsilon": float(self.agent.epsilon),
            "exploration": float(self.agent.exploration_value),
            "train_metrics": train_metrics,
        }
        self.history.append(meta)
        return new_global, meta


def build_dataset_and_users(args: Any):
    if args.dataset == "mnist":
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset_train = datasets.MNIST(
            "../data/mnist/", train=True, download=True, transform=trans_mnist
        )
        dataset_test = datasets.MNIST(
            "../data/mnist/", train=False, download=True, transform=trans_mnist
        )
        dict_users = (
            mnist_iid(dataset_train, args.num_users)
            if args.iid
            else mnist_noniid(dataset_train, args.num_users)
        )
    elif args.dataset == "fashion_mnist":
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])]
        )
        dataset_train = datasets.FashionMNIST(
            "../data/", train=True, download=True, transform=trans_mnist
        )
        dataset_test = datasets.FashionMNIST(
            "../data/", train=False, download=True, transform=trans_mnist
        )
        if args.iid:
            dict_users = np.load("./data/iid_fashion_mnist.npy", allow_pickle=True).item()
        else:
            dict_users = np.load("./data/non_iid_fashion_mnist.npy", allow_pickle=True).item()
    elif args.dataset == "cifar":
        trans_cifar = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset_train = datasets.CIFAR10(
            "../data/cifar", train=True, download=True, transform=trans_cifar
        )
        dataset_test = datasets.CIFAR10(
            "../data/cifar", train=False, download=True, transform=trans_cifar
        )
        if args.iid:
            dict_users = np.load("./data/iid_cifar.npy", allow_pickle=True).item()
        else:
            dict_users = cifar_noniid(
                [x[1] for x in dataset_train], args.num_users, 10, args.p
            )
    else:
        raise ValueError("rule_rl runner currently supports mnist, fashion_mnist, cifar")
    return dataset_train, dataset_test, dict_users


def build_model(args: Any):
    if args.model == "VGG" and args.dataset == "cifar":
        return vgg19_bn().to(args.device)
    if args.model == "VGG11" and args.dataset == "cifar":
        return vgg11().to(args.device)
    if args.model == "resnet" and args.dataset == "cifar":
        return ResNet18().to(args.device)
    if args.model == "resnet20" and args.dataset == "cifar":
        return resnet20().to(args.device)
    if args.model == "rlr_mnist" or args.model == "cnn":
        return get_model("fmnist").to(args.device)
    raise ValueError("rule_rl runner requires an image model with get_feature")


def separability_auc(values: Sequence[float], labels: Sequence[int]) -> float:
    """AUC = P(value on attack round > value on clean round).

    0.5 = no separation (the metric is blind to the attack); ->1.0 = metric is
    higher on attack rounds; ->0.0 = lower on attack rounds. The *strength* of
    separation is |AUC - 0.5| * 2 regardless of direction.
    """
    pos = [float(v) for v, l in zip(values, labels) if l == 1 and np.isfinite(v)]
    neg = [float(v) for v, l in zip(values, labels) if l == 0 and np.isfinite(v)]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def summarize_diagnostics(
    diag_records: Sequence[Tuple[int, Dict[str, float]]],
    wandb_run: Any = None,
) -> None:
    """Decide whether the trigger-free signal separates attack vs clean rounds.

    For every diagnostic key we report clean-round mean, attack-round mean and
    the separation AUC, then log them to the wandb run summary. This is the
    artifact that answers Q1 (does the signal react to attacks?) and, via the
    ``delta_*`` keys, whether our filter actually removes the anomaly.
    """
    if not diag_records:
        print("[diag] no diagnostic records collected (diagnostics disabled?)")
        return

    labels = [1 if count > 0 else 0 for count, _ in diag_records]
    n_attack = sum(labels)
    n_clean = len(labels) - n_attack
    all_keys = sorted({k for _, d in diag_records for k in d})
    oracle_keys = [k for k in all_keys if k.startswith("oraclegap_")]
    cross_keys = [k for k in all_keys if not k.startswith("oraclegap_")]

    print("\n==== Diagnostic summary ====")
    print(f"rounds total={len(labels)}  attack={n_attack}  clean={n_clean}")

    # --- Q1 (confound-free): within-round oracle gap = anomaly(all) -
    #     anomaly(benign-only), over attack rounds. Positive & consistent means
    #     malicious presence really moves the clean signal. ---
    if oracle_keys:
        print("\n[oracle within-round gap]  (anomaly_with_mal - anomaly_without_mal)")
        print(f"{'metric':24s}{'mean_gap':>12s}{'frac>0':>9s}  verdict")
        for k in oracle_keys:
            gaps = [d[k] for _, d in diag_records if k in d and np.isfinite(d[k])]
            if not gaps:
                continue
            mean_gap = float(np.mean(gaps))
            frac_pos = float(np.mean([1.0 if g > 0 else 0.0 for g in gaps]))
            consistency = max(frac_pos, 1.0 - frac_pos)  # how one-directional
            verdict = "STRONG" if consistency >= 0.8 and abs(mean_gap) > 1e-6 else (
                "weak" if consistency >= 0.65 else "none")
            print(f"{k[len('oraclegap_'):]:24s}{mean_gap:12.4f}{frac_pos:9.2f}  {verdict}")
            if wandb_run is not None:
                wandb_run.summary[f"diag_oracle/{k}_mean_gap"] = mean_gap
                wandb_run.summary[f"diag_oracle/{k}_frac_pos"] = frac_pos

    # --- Cross-round attack-vs-clean AUC (cheap, but maturity-confounded:
    #     clean rounds tend to be earlier/less mature). Secondary view. ---
    if cross_keys and n_attack > 0 and n_clean > 0:
        print("\n[cross-round attack vs clean]  (maturity-confounded, secondary)")
        print(f"{'metric':30s}{'clean_mean':>12s}{'attack_mean':>12s}{'AUC':>8s}  verdict")
        for k in cross_keys:
            vals = [d.get(k, float('nan')) for _, d in diag_records]
            clean_v = [v for v, l in zip(vals, labels) if l == 0 and np.isfinite(v)]
            attack_v = [v for v, l in zip(vals, labels) if l == 1 and np.isfinite(v)]
            cm = float(np.mean(clean_v)) if clean_v else float('nan')
            am = float(np.mean(attack_v)) if attack_v else float('nan')
            auc = separability_auc(vals, labels)
            strength = abs(auc - 0.5) * 2 if np.isfinite(auc) else 0.0
            verdict = "STRONG" if strength >= 0.6 else ("weak" if strength >= 0.3 else "none")
            print(f"{k:30s}{cm:12.4f}{am:12.4f}{auc:8.3f}  {verdict}")
            if wandb_run is not None:
                wandb_run.summary[f"diag_sep/{k}_auc"] = auc
                wandb_run.summary[f"diag_sep/{k}_clean_mean"] = cm
                wandb_run.summary[f"diag_sep/{k}_attack_mean"] = am
    elif cross_keys:
        print("\n[cross-round] need BOTH attack and clean rounds; skipped (use oracle gap above)")
    print("============================\n")


def summarize_per_client(
    records: Sequence[Dict[str, Any]],
    wandb_run: Any = None,
) -> None:
    """Are malicious clients outliers among individual clients on clean data?

    This is the test that matters for stage-2: the bandit's rules filter
    per-client in embedding space, so we ask whether malicious clients are
    separable there (``emb_dist``) and/or in their own clean confidence. AUC =
    P(malicious value > benign value); strength = |AUC - 0.5| * 2.
    """
    if not records:
        print("[per-client] no records (no attack rounds, or diagnostics off)")
        return
    labels = [1 if r.get("malicious") else 0 for r in records]
    n_mal = sum(labels)
    n_ben = len(labels) - n_mal
    print("\n==== Per-client separability: malicious vs benign (same-round) ====")
    print(f"client-rounds total={len(labels)}  malicious={n_mal}  benign={n_ben}")
    if n_mal == 0 or n_ben == 0:
        print("[per-client] need both malicious and benign client-rounds; skipping")
        return
    keys = sorted({k for r in records for k in r if k != "malicious"})
    print(f"{'metric':18s}{'benign_mean':>13s}{'malic_mean':>13s}{'AUC':>8s}  verdict")
    for k in keys:
        vals = [r.get(k, float('nan')) for r in records]
        bm = float(np.nanmean([v for v, l in zip(vals, labels) if l == 0]))
        mm = float(np.nanmean([v for v, l in zip(vals, labels) if l == 1]))
        auc = separability_auc(vals, labels)
        strength = abs(auc - 0.5) * 2 if np.isfinite(auc) else 0.0
        verdict = "STRONG" if strength >= 0.6 else ("weak" if strength >= 0.3 else "none")
        print(f"{k:18s}{bm:13.4f}{mm:13.4f}{auc:8.3f}  {verdict}")
        if wandb_run is not None:
            wandb_run.summary[f"per_client/{k}_auc"] = auc
            wandb_run.summary[f"per_client/{k}_benign_mean"] = bm
            wandb_run.summary[f"per_client/{k}_malicious_mean"] = mm
    print("===================================================================\n")


@dataclass
class TriggerAction:
    """A probe trigger: a white square patch of ``size`` at (py, px)."""
    px: int
    py: int
    size: int


def build_trigger_actions(h: int, w: int, sizes: Tuple[int, ...], step: int) -> List[TriggerAction]:
    actions: List[TriggerAction] = []
    for size in sizes:
        ys = list(range(0, max(h - size, 0) + 1, step))
        xs = list(range(0, max(w - size, 0) + 1, step))
        if (h - size) not in ys:
            ys.append(max(h - size, 0))
        if (w - size) not in xs:
            xs.append(max(w - size, 0))
        for py in ys:
            for px in xs:
                actions.append(TriggerAction(px=int(px), py=int(py), size=int(size)))
    return actions


def otsu_split(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """Best 1-D split (Otsu) -- magnitude-aware.

    Returns (threshold, bcv, gap, bimodality):
      bcv = w_low * w_high * (mean_high - mean_low)^2  (raw between-group var)
      gap = mean_high - mean_low                       (absolute separation)
      bimodality = bcv / total_var in [0,1]            (scale-free; saturates)
    bcv is the reward signal: it rewards a *minority* group (w_low*w_high peaks at
    a balanced split) that shifts by a *large* amount (gap). Normalized bimodality
    saturates on 10 points (any tiny clean split -> 1.0), so it is logged only.
    """
    v = np.sort(np.asarray(list(values), dtype=np.float64))
    n = v.size
    if n < 2 or np.allclose(v, v[0]):
        return (float(v[-1]) if n else 0.0), 0.0, 0.0, 0.0
    total_var = float(np.var(v)) + EPS
    best_bcv = -1.0
    best_thr = float(v[-1])
    best_gap = 0.0
    for i in range(1, n):
        w0 = i / n
        w1 = 1.0 - w0
        m0 = float(v[:i].mean())
        m1 = float(v[i:].mean())
        bcv = w0 * w1 * (m1 - m0) ** 2
        if bcv > best_bcv:
            best_bcv = bcv
            best_thr = float((v[i - 1] + v[i]) / 2.0)
            best_gap = m1 - m0
    return best_thr, float(best_bcv), float(best_gap), float(best_bcv / total_var)


class TriggerProbeServer:
    """Stage-2 by trigger probing instead of clean-embedding filtering.

    Each action is a probe trigger. We stamp it on the clean validation set,
    run every client model, and measure per-client susceptibility = how much the
    trigger shifts predictions toward *some* class (max class-shift; the target
    is discovered, not enumerated). A backdoored client whose planted trigger
    matches the probe shifts strongly; benign clients barely move. The label-free
    reward is the bimodality (Otsu) of susceptibility across clients: a good probe
    splits clients into responders / non-responders. Ground-truth AUC is logged
    only for validation -- it never feeds the reward.
    """

    def __init__(self, args: Any, validation_dataset: Any, validation_indices: Sequence[int]):
        self.args = args
        self.validation_dataset = validation_dataset
        self.validation_indices = list(validation_indices)
        img = validation_dataset[self.validation_indices[0]][0]
        h, w = int(img.shape[-2]), int(img.shape[-1])
        sizes = tuple(int(s) for s in getattr(args, "trigger_probe_sizes", (5,)))
        step = int(getattr(args, "trigger_probe_step", 6))
        self.actions = build_trigger_actions(h, w, sizes, step)
        # Shuffle probe order (seeded) so the true trigger is NOT at arm 0: the
        # bandit's cold-start argmax picks index 0, so an unshuffled grid would
        # let a corner (0,0) trigger be "found" for free. Shuffling forces a real
        # search. Order is irrelevant to the deployed defense.
        np.random.RandomState(int(args.seed) + 12345).shuffle(self.actions)
        config = BanditConfig(
            state_dim=1,
            num_actions=len(self.actions),
            mode=getattr(args, "rule_bandit_mode", "LinUCB"),
            lr=float(getattr(args, "rule_bandit_lr", 1e-3)),
            batch_size=int(getattr(args, "rule_bandit_batch_size", 16)),
            replay_capacity=5000,
            lambda_reg=float(getattr(args, "rule_bandit_lambda", 1.0)),
            alpha=float(getattr(args, "rule_bandit_alpha", 1.0)),
            nu=float(getattr(args, "rule_bandit_nu", 1.0)),
            hidden_dim=int(getattr(args, "rule_bandit_hidden_dim", 128)),
            feature_dim=int(getattr(args, "rule_bandit_feature_dim", 64)),
            train_steps_per_update=int(getattr(args, "rule_bandit_train_steps", 1)),
            seed=args.seed,
            device=str(args.device),
        )
        self.agent = create_rule_bandit_agent(actions=self.actions, config=config)
        # (b) true-bandit: probe only a small budget of triggers per round and
        # learn which region exposes the (fixed, unknown) attacker trigger across
        # rounds. 0 => exhaustive (evaluate all probes; validation/baseline).
        self.evals_per_round = int(getattr(args, "trigger_probe_evals", 0))
        # (c) stage-1 update-space prefilter (multi-Krum) before trigger probing
        self.use_stage1 = bool(getattr(args, "trigger_stage1", True))
        for attr in ("turn", "wrong_mal", "mal_score", "ben_score"):
            if not hasattr(args, attr):
                setattr(args, attr, 0)
        self.history: List[Dict[str, Any]] = []
        self._scratch_net: Optional[torch.nn.Module] = None
        self._val_batches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    def _get_scratch_net(self, net_template: torch.nn.Module) -> torch.nn.Module:
        if self._scratch_net is None:
            self._scratch_net = copy.deepcopy(net_template).to(self.args.device)
        return self._scratch_net

    def _get_val_batches(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if self._val_batches is None:
            loader = DataLoader(
                DatasetSplit(self.validation_dataset, self.validation_indices),
                batch_size=self.args.bs,
                shuffle=False,
            )
            self._val_batches = [
                (data.to(self.args.device), target.to(self.args.device))
                for data, target in loader
            ]
        return self._val_batches

    @staticmethod
    def _stamp(images: torch.Tensor, action: TriggerAction) -> torch.Tensor:
        out = images.clone()
        val = float(images.max())
        out[:, :, action.py:action.py + action.size, action.px:action.px + action.size] = val
        return out

    @torch.no_grad()
    def _susceptibility_matrix(
        self,
        net_template: torch.nn.Module,
        local_weights: Sequence[Dict[str, torch.Tensor]],
        action_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """(n_clients, n_probes, n_classes) per-class prediction shift.

        shift[i, j, c] = P(pred=c | probe j) - P(pred=c | clean) for client i,
        where probe j = ``action_ids[j]`` (default: all actions). We keep the
        *target* dimension (do not collapse with max) so the reward can demand a
        consistent target: a backdoor trigger pulls a subset of clients toward the
        SAME class, which only shows up per-class. Passing a subset of action_ids
        lets the bandit probe only a few triggers per round (amortized search).
        """
        if action_ids is None:
            action_ids = list(range(len(self.actions)))
        probes = [self.actions[a] for a in action_ids]
        net = self._get_scratch_net(net_template)
        batches = self._get_val_batches()
        num_probes = len(probes)
        shift: Optional[np.ndarray] = None
        for ci, w in enumerate(local_weights):
            net.load_state_dict(w)
            net.eval()
            num_classes = None
            clean_counts = None
            total = 0
            trig_counts = [None] * num_probes
            for data, _ in batches:
                clean_logits = net(data)
                if num_classes is None:
                    num_classes = clean_logits.size(1)
                    clean_counts = np.zeros(num_classes)
                    trig_counts = [np.zeros(num_classes) for _ in range(num_probes)]
                    if shift is None:
                        shift = np.zeros((len(local_weights), num_probes, num_classes))
                total += data.size(0)
                cp = clean_logits.argmax(1).cpu().numpy()
                for c in cp:
                    clean_counts[c] += 1
                for j, action in enumerate(probes):
                    tp = net(self._stamp(data, action)).argmax(1).cpu().numpy()
                    for c in tp:
                        trig_counts[j][c] += 1
            clean_frac = clean_counts / max(total, 1)
            for j in range(num_probes):
                shift[ci, j, :] = trig_counts[j] / max(total, 1) - clean_frac
        return shift if shift is not None else np.zeros((len(local_weights), num_probes, 1))

    def _stage1_prefilter(self, local_updates: Sequence[Dict[str, torch.Tensor]]) -> List[int]:
        """Update-space multi-Krum prefilter (skipped when too few benign)."""
        n = len(local_updates)
        if not self.use_stage1 or local_updates is None or n == 0:
            return list(range(n))
        num_clients = max(int(self.args.frac * self.args.num_users), 1)
        f = max(int(self.args.malicious * num_clients), 1)
        if n <= 2 * f + 2:  # multi-Krum needs a benign majority; otherwise keep all
            return list(range(n))
        kept = [int(i) for i in multi_krum(list(local_updates), f, self.args, multi_k=True)]
        return sorted(set(kept)) if kept else list(range(n))

    def _probe_scores(self, shift_col: np.ndarray):
        """Given (n_clients, n_classes) shift for one probe, return
        (reward=bcv, target_class, threshold, gap, susceptibility_column)."""
        num_classes = shift_col.shape[1]
        stats = [otsu_split(shift_col[:, c]) for c in range(num_classes)]
        bcvs = [s[1] for s in stats]
        bc = int(np.argmax(bcvs))
        thr, bcv, gap, _bimo = stats[bc]
        return float(bcv), bc, float(thr), float(gap), shift_col[:, bc]

    def aggregate(
        self,
        round_idx: int,
        net_template: torch.nn.Module,
        local_weights: Sequence[Dict[str, torch.Tensor]],
        local_updates: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
        n_malicious: int = 0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        n = len(local_weights)
        state = np.array([1.0], dtype=np.float32)
        labels = [1 if i < n_malicious else 0 for i in range(n)]

        # (c) stage-1: update-space multi-Krum prefilter
        kept1 = self._stage1_prefilter(local_updates if local_updates is not None else [])
        # stage-2 only probes the stage-1 survivors (no point probing clients we
        # already rejected; also cheaper). Work in survivor-local indices, map
        # back to original client ids at the end.
        surv_weights = [local_weights[i] for i in kept1]
        surv_labels = [labels[i] for i in kept1]
        ns = len(kept1)
        n_mal_surv = int(sum(surv_labels))

        # (b) choose which probe triggers to evaluate this round. With a budget,
        # the bandit explores a few (UCB) + commits greedy; over rounds it locks
        # onto the region where the fixed attacker trigger lives. evals==0 ->
        # exhaustive (validation baseline).
        if self.evals_per_round and self.evals_per_round < len(self.actions):
            # UCB top-k over probes: mean reward (from the bandit) + a count-based
            # exploration bonus. select_action is deterministic, so we score arms
            # ourselves to get DISTINCT probes; unseen arms get a high bonus so the
            # search sweeps the space, then concentrates where reward is high.
            means = np.asarray(self.agent.get_q_values(state), dtype=np.float64)
            counts = np.asarray(getattr(self.agent, "action_counts",
                                        np.zeros(len(self.actions))), dtype=np.float64)
            t = float(counts.sum()) + 1.0
            alpha = float(getattr(self.args, "rule_bandit_alpha", 1.0))
            ucb = means + alpha * np.sqrt(np.log(t + 1.0) / (counts + 1.0))
            eval_ids = [int(i) for i in np.argsort(-ucb)[: self.evals_per_round]]
            g = int(self.agent.select_action(state, eval_mode=True))  # greedy commit candidate
            if g not in eval_ids:
                eval_ids.append(g)
        else:
            eval_ids = list(range(len(self.actions)))

        # too few survivors to split -> stage-2 inactive, keep stage-1 result
        if ns < 2:
            new_global = FedAvg([local_weights[i] for i in kept1]) if kept1 else FedAvg(list(local_weights))
            act0 = self.actions[eval_ids[0]]
            meta = {
                "round": int(round_idx), "num_clients": int(n),
                "kept_clients": [int(i) for i in kept1],
                "stage1_kept": [int(i) for i in kept1],
                "best_action": {"px": act0.px, "py": act0.py, "size": act0.size, "target": -1},
                "diag": {"n_probes_evaluated": 0, "stage1_removed": int(n - ns),
                         "stage1_mal_removed": int(n_malicious - n_mal_surv) if n_malicious else 0,
                         "reward_best": 0.0, "gap_best": 0.0,
                         "auc_at_best_bimo": float("nan"), "auc_oracle": float("nan"),
                         "n_flagged": 0, "mal_flagged": 0, "ben_flagged": 0,
                         "mal_survive": n_mal_surv, "ben_survive": int(ns - n_mal_surv),
                         "mal_total": int(n_malicious), "ben_total": int(n - n_malicious)},
            }
            self.history.append(meta)
            return new_global, meta

        # probe ONLY the survivors
        shift = self._susceptibility_matrix(net_template, surv_weights, eval_ids)  # (ns, P, C)

        rewards = np.zeros(len(eval_ids))
        gaps = np.zeros(len(eval_ids))
        thrs = np.zeros(len(eval_ids))
        tgts = np.zeros(len(eval_ids), dtype=int)
        cols = np.zeros((ns, len(eval_ids)))
        gt_auc = np.full(len(eval_ids), np.nan)
        for j, ai in enumerate(eval_ids):
            bcv, bc, thr_j, gap_j, col = self._probe_scores(shift[:, j, :])
            rewards[j], tgts[j], thrs[j], gaps[j], cols[:, j] = bcv, bc, thr_j, gap_j, col
            if 0 < n_mal_surv < ns:
                gt_auc[j] = separability_auc(col.tolist(), surv_labels)
            self.agent.push_transition(state, ai, float(bcv), state, True)
        self.agent.update()

        best_j = int(np.argmax(rewards))      # commit by label-free reward (bcv)
        best_ai = eval_ids[best_j]
        col = cols[:, best_j]
        flagged = col > thrs[best_j]          # boolean over survivors (local index)
        active = gaps[best_j] >= 0.1          # magnitude floor: act only on a real signal

        # (c) compose: among stage-1 survivors, drop the trigger responders.
        if active:
            keep = [kept1[s] for s in range(ns) if not flagged[s]]
        else:
            keep = list(kept1)
        if len(keep) == 0:
            keep = list(kept1)
        new_global = FedAvg([local_weights[i] for i in keep])

        oracle_j = int(np.nanargmax(gt_auc)) if np.any(np.isfinite(gt_auc)) else best_j
        act = self.actions[best_ai]
        meta: Dict[str, Any] = {
            "round": int(round_idx),
            "num_clients": int(n),
            "kept_clients": [int(i) for i in keep],
            "stage1_kept": [int(i) for i in kept1],
            "best_action": {"px": act.px, "py": act.py, "size": act.size, "target": int(tgts[best_j])},
            "diag": {
                "n_probes_evaluated": int(len(eval_ids)),
                "stage1_removed": int(n - ns),
                "stage1_mal_removed": int(n_malicious - n_mal_surv) if n_malicious else 0,
                "reward_best": float(rewards[best_j]),
                "gap_best": float(gaps[best_j]),
                "auc_at_best_bimo": float(gt_auc[best_j]) if np.isfinite(gt_auc[best_j]) else float("nan"),
                "auc_oracle": float(gt_auc[oracle_j]) if np.isfinite(gt_auc[oracle_j]) else float("nan"),
                "n_flagged": int(np.sum(flagged)) if active else 0,
                "mal_flagged": int(np.sum([flagged[s] for s in range(ns) if surv_labels[s] == 1])) if (n_mal_surv and active) else 0,
                "ben_flagged": int(np.sum([flagged[s] for s in range(ns) if surv_labels[s] == 0])) if active else 0,
                # final composition (lower mal_survive / higher ben_survive = better)
                "mal_survive": int(sum(1 for i in keep if labels[i] == 1)) if n_malicious else 0,
                "ben_survive": int(sum(1 for i in keep if labels[i] == 0)),
                "mal_total": int(n_malicious),
                "ben_total": int(n - n_malicious),
            },
        }
        self.history.append(meta)
        return new_global, meta


def summarize_trigger_probe(
    records: Sequence[Tuple[int, Dict[str, float]]],
    wandb_run: Any = None,
) -> None:
    """Does the label-free probe (max-bimodality trigger) actually separate
    malicious, and how close is it to the oracle ceiling? Over attack rounds."""
    attack = [d for c, d in records if c > 0 and d]
    if not attack:
        print("[trigger-probe] no attack rounds with diagnostics")
        return

    def mean(rows, key):
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    half = max(len(attack) // 2, 1)
    early, late = attack[:half], attack[half:]
    # detection rate = malicious caught / malicious present (stage-1 + stage-2)
    def detect_rate(rows):
        caught = [(r.get("mal_total", 0) - r.get("mal_survive", 0)) for r in rows]
        tot = [r.get("mal_total", 0) for r in rows]
        s = sum(tot)
        return (sum(caught) / s) if s else float("nan")
    def ben_keep_rate(rows):
        kept = [r.get("ben_survive", 0) for r in rows]
        tot = [r.get("ben_total", 0) for r in rows]
        s = sum(tot)
        return (sum(kept) / s) if s else float("nan")

    print("\n==== Trigger-probe summary (attack rounds) ====")
    print(f"attack rounds={len(attack)}  mean probes/round={mean(attack, 'n_probes_evaluated'):.1f}"
          f"  (total triggers in space, see launch log)")
    print(f"AUC @ committed probe (label-free): {mean(attack, 'auc_at_best_bimo'):.3f}"
          f"   [oracle among probed: {mean(attack, 'auc_oracle'):.3f}]")
    print(f"malicious detection rate  overall {detect_rate(attack):.2f} | "
          f"early {detect_rate(early):.2f} -> late {detect_rate(late):.2f}  (learning if late>early)")
    print(f"benign retention rate     overall {ben_keep_rate(attack):.2f} | "
          f"early {ben_keep_rate(early):.2f} -> late {ben_keep_rate(late):.2f}")
    print(f"stage-1 removed/round {mean(attack, 'stage1_removed'):.2f} "
          f"(malicious {mean(attack, 'stage1_mal_removed'):.2f}) | "
          f"stage-2 flagged/round mal {mean(attack, 'mal_flagged'):.2f}/ben {mean(attack, 'ben_flagged'):.2f}")
    print("verdict: works if detection high & benign retention high; RL learns if late>early")
    print("===============================================\n")
    if wandb_run is not None:
        wandb_run.summary["trigger_probe/auc_committed"] = mean(attack, "auc_at_best_bimo")
        wandb_run.summary["trigger_probe/detect_rate"] = detect_rate(attack)
        wandb_run.summary["trigger_probe/detect_rate_early"] = detect_rate(early)
        wandb_run.summary["trigger_probe/detect_rate_late"] = detect_rate(late)
        wandb_run.summary["trigger_probe/benign_retention"] = ben_keep_rate(attack)
        wandb_run.summary["trigger_probe/probes_per_round"] = mean(attack, "n_probes_evaluated")


def main() -> None:
    args = args_parser()
    args.defence = "rule_rl"
    if args.attack == "lp_attack":
        args.attack = "adaptive"
    args.device = torch.device(
        "cuda:{}".format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else "cpu"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    test_mkdir("./" + args.save)
    print_exp_details(args)

    dataset_train, dataset_test, dict_users = build_dataset_and_users(args)
    net_glob = build_model(args)
    if not hasattr(net_glob, "get_feature"):
        raise AttributeError("Selected model does not expose get_feature(images)")
    net_glob.train()

    w_glob = net_glob.state_dict()
    if args.init != "None":
        param = torch.load(args.init, map_location=args.device)
        net_glob.load_state_dict(param)
        w_glob = net_glob.state_dict()
        print("load init model")

    if math.isclose(args.malicious, 0):
        backdoor_begin_acc = 100
    else:
        backdoor_begin_acc = args.attack_begin

    central_dataset = central_dataset_iid(dataset_test, args.server_dataset)
    val_idx_set = set(int(i) for i in central_dataset)
    eval_indices = [i for i in range(len(dataset_test)) if i not in val_idx_set]
    dataset_eval = Subset(dataset_test, eval_indices)
    trigger_probe = bool(getattr(args, "trigger_probe", False))
    if trigger_probe:
        rule_rl_server = TriggerProbeServer(args, dataset_test, central_dataset)
        print(f"[trigger-probe] {len(rule_rl_server.actions)} probe triggers")
    else:
        rule_rl_server = RuleRLServer(args, dataset_test, central_dataset)

    base_info = get_base_info(args)
    filename = "./" + args.save + "/accuracy_file_{}.txt".format(base_info)
    wandb_run = init_wandb(args, base_info, len(central_dataset), len(eval_indices))

    val_acc_list = [0.0001]
    backdoor_acculist = [0]
    loss_train = []
    diag_records: List[Tuple[int, Dict[str, float]]] = []  # (attack_count, diag) per round
    per_client_records: List[Dict[str, Any]] = []  # per-client outlierness on attack rounds
    args.attack_layers = []
    if args.attack == "dba":
        args.dba_sign = 0

    malicious_list = [i for i in range(int(args.num_users * args.malicious))]

    for iter_idx in range(args.epochs):
        loss_locals = []
        round_w_locals = []
        round_w_updates = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        if backdoor_begin_acc < val_acc_list[-1]:
            backdoor_begin_acc = 0
            attack_number = int(args.malicious * m)
        else:
            attack_number = 0

        if args.scaling_attack_round != 1:
            if iter_idx > 100 and iter_idx % args.scaling_attack_round == 0:
                attack_number = attack_number
            else:
                attack_number = 0

        round_attack_count = attack_number

        for _, idx in enumerate(idxs_users):
            if attack_number > 0:
                args.iter = iter_idx
                mal_weight, loss, args.attack_layers = attacker(
                    malicious_list,
                    attack_number,
                    args.attack,
                    dataset_train,
                    dataset_test,
                    dict_users,
                    net_glob,
                    args,
                    idx=None,
                )
                attack_number -= 1
                w = mal_weight[0]
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            round_w_updates.append(get_update(w, w_glob))
            round_w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        if trigger_probe:
            w_glob, rule_rl_meta = rule_rl_server.aggregate(
                iter_idx, net_glob, round_w_locals, round_w_updates,
                n_malicious=round_attack_count,
            )
        else:
            w_glob, rule_rl_meta = rule_rl_server.aggregate(
                iter_idx, net_glob, round_w_locals, round_w_updates,
                n_malicious=round_attack_count,
            )
        # tag the round's attack ground-truth onto meta (also flows into history)
        # and record (attack_count, diag) for the end-of-run separability summary
        rule_rl_meta["round_attack_count"] = int(round_attack_count)
        diag_records.append((int(round_attack_count), dict(rule_rl_meta.get("diag") or {})))
        per_client_records.extend(rule_rl_meta.get("per_client") or [])
        if trigger_probe:
            d = rule_rl_meta["diag"]
            ba = rule_rl_meta["best_action"]
            print(
                "TriggerProbe round {:3d}: probes {} | probe(px={},py={},size={},tgt={}) | gap {:.3f} | "
                "auc@reward {:.3f} | s1_rm {} | flagged(mal {}/ben {}) | survive mal {}/{} ben {}/{}".format(
                    iter_idx, d["n_probes_evaluated"], ba["px"], ba["py"], ba["size"], ba.get("target", -1),
                    d["gap_best"], d["auc_at_best_bimo"], d["stage1_removed"],
                    d["mal_flagged"], d["ben_flagged"],
                    d["mal_survive"], d["mal_total"], d["ben_survive"], d["ben_total"],
                )
            )
        else:
            action = rule_rl_meta["action"]
            print(
                "RuleBandit[{}] round {:3d}: unique/exec {}/{} | prefilter {}/{} | selected {} | "
                "commit action {}({},{}) | val_acc {:.2f} | sep {:.3f} | reward {:.4f} | "
                "rollout_mean {:.4f} | rollout_best {:.4f} | explore {:.3f}".format(
                    rule_rl_meta["bandit_mode"],
                    iter_idx,
                    rule_rl_meta["num_transitions_added"],
                    rule_rl_meta["num_rollouts_executed"],
                    len(rule_rl_meta["prefiltered_clients"]),
                    rule_rl_meta["num_clients"],
                    rule_rl_meta["selected_clients"],
                    rule_rl_meta["action_id"],
                    action["score_type"],
                    action["center_type"],
                    rule_rl_meta["validation_accuracy"],
                    rule_rl_meta["separation"],
                    rule_rl_meta["reward"],
                    rule_rl_meta["rollout_reward_mean"],
                    rule_rl_meta["rollout_reward_max"],
                    rule_rl_meta["exploration"],
                )
            )

        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print("Round {:3d}, Average loss {:.3f}".format(iter_idx, loss_avg))
        loss_train.append(loss_avg)

        acc_test, _, back_acc = test_img(net_glob, dataset_eval, args, test_backdoor=True)
        main_acc = to_float(acc_test)
        backdoor_acc = to_float(back_acc)
        print("Main accuracy: {:.2f}".format(main_acc))
        print("Backdoor accuracy: {:.2f}".format(backdoor_acc))
        val_acc_list.append(main_acc)
        backdoor_acculist.append(backdoor_acc)
        if wandb_run is not None:
            if trigger_probe:
                metrics = {
                    "round": iter_idx,
                    "train/loss_avg": to_float(loss_avg),
                    "eval/main_accuracy": main_acc,
                    "eval/backdoor_accuracy": backdoor_acc,
                    "attack/active": float(round_attack_count > 0),
                    "attack/round_malicious_clients": int(round_attack_count),
                }
                for k, v in (rule_rl_meta.get("diag") or {}).items():
                    metrics[f"trigger_probe/{k}"] = to_float(v)
                wandb_run.log(metrics, step=iter_idx)
            else:
                wandb_run.log(
                    build_wandb_metrics(
                        iter_idx, loss_avg, main_acc, backdoor_acc, rule_rl_meta, round_attack_count
                    ),
                    step=iter_idx,
                )
        write_file(filename, val_acc_list, backdoor_acculist, args)

    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)
    torch.save(rule_rl_server.history, "./" + args.save + "/rule_rl_history.pt")

    # Q1 + delta verdict: does the trigger-free signal react to attacks, and
    # does our filter remove the anomaly? Printed and pushed to wandb summary.
    if trigger_probe:
        summarize_trigger_probe(diag_records, wandb_run)
    else:
        summarize_diagnostics(diag_records, wandb_run)
        summarize_per_client(per_client_records, wandb_run)

    plt.figure()
    plt.xlabel("communication")
    plt.ylabel("accu_rate")
    plt.plot(val_acc_list, label="main task(acc:" + str(best_acc) + "%)")
    plt.plot(
        backdoor_acculist,
        label="backdoor task(BBSR:" + str(bbsr) + "%, ABSR:" + str(absr) + "%)",
    )
    plt.legend()
    title = base_info
    plt.title(title)
    plt.savefig("./" + args.save + "/" + title + ".pdf", format="pdf", bbox_inches="tight")

    net_glob.eval()
    acc_train, _ = test_img(net_glob, dataset_train, args)
    acc_test, _ = test_img(net_glob, dataset_eval, args)
    final_train_acc = to_float(acc_train)
    final_test_acc = to_float(acc_test)
    print("Training accuracy: {:.2f}".format(final_train_acc))
    print("Testing accuracy: {:.2f}".format(final_test_acc))

    if wandb_run is not None:
        wandb_run.log(
            {
                "final/train_accuracy": final_train_acc,
                "final/test_accuracy": final_test_acc,
                "final/best_main_accuracy": to_float(best_acc),
                "final/absr": to_float(absr),
                "final/bbsr": to_float(bbsr),
            },
            step=args.epochs,
        )
        wandb_run.summary["best_main_accuracy"] = to_float(best_acc)
        wandb_run.summary["absr"] = to_float(absr)
        wandb_run.summary["bbsr"] = to_float(bbsr)
        wandb_run.summary["final_train_accuracy"] = final_train_acc
        wandb_run.summary["final_test_accuracy"] = final_test_acc
        wandb_run.finish()

    torch.save(net_glob.state_dict(), "./" + args.save + "/model.pth")


if __name__ == "__main__":
    main()
