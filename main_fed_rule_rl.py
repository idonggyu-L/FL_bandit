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
        self.last_validation_accuracy: Optional[float] = None
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
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if len(local_weights) == 0:
            raise ValueError("No local weights were provided for aggregation")

        embeddings = self.collect_client_embeddings(net_template, local_weights)
        update_norms = [update_l2_norm(update) for update in local_updates]
        prefiltered = self.filter.select(self.args, local_updates, embeddings)

        state = build_rule_state(embeddings, prefiltered, update_norms)

        if self.last_validation_accuracy is None:
            self.last_validation_accuracy = self.validation_accuracy(net_template)
        baseline_accuracy = float(self.last_validation_accuracy)

        rollout_records: List[Dict[str, Any]] = []
        train_metrics: Optional[Dict[str, float]] = None
        action_cache: Dict[int, Dict[str, Any]] = {}
        num_actions = len(self.agent.actions)
        for rollout_idx in range(self.rollouts_per_round):
            action_id = self.agent.select_action(state)
            action = self.agent.get_action_object(action_id)

            if action_id in action_cache:
                cached = action_cache[action_id]
                selected_clients = cached["selected_clients"]
                candidate_accuracy = cached["validation_accuracy"]
                reward = cached["reward"]
                is_new = False
            else:
                selected_clients = self.select_clients_for_action(
                    embeddings, prefiltered, action
                )
                candidate_global = FedAvg([local_weights[i] for i in selected_clients])
                candidate_accuracy = self.validation_accuracy(net_template, candidate_global)
                reward = float((candidate_accuracy - baseline_accuracy) / 100.0)
                action_cache[action_id] = {
                    "selected_clients": selected_clients,
                    "validation_accuracy": candidate_accuracy,
                    "reward": reward,
                }
                is_new = True

                self.agent.push_transition(state, action_id, reward, state, True)
                updated_metrics = self.agent.update()
                if updated_metrics is not None:
                    train_metrics = updated_metrics

            rollout_records.append(
                {
                    "rollout_idx": int(rollout_idx),
                    "action_id": int(action_id),
                    "action": {
                        "score_type": action.score_type,
                        "center_type": action.center_type,
                        "drop_inner_ratio": float(action.drop_inner_ratio),
                        "keep_outer_ratio": float(action.keep_outer_ratio),
                        "knn_k": int(action.knn_k),
                    },
                    "selected_clients": [int(i) for i in selected_clients],
                    "reward": reward,
                    "validation_accuracy": float(candidate_accuracy),
                    "is_new": bool(is_new),
                }
            )

            if len(action_cache) >= num_actions:
                break

        final_action_id = self.agent.select_action(state, eval_mode=True)
        final_action = self.agent.get_action_object(final_action_id)
        final_selected_clients = self.select_clients_for_action(
            embeddings, prefiltered, final_action
        )
        new_global = FedAvg([local_weights[i] for i in final_selected_clients])
        final_accuracy = self.validation_accuracy(net_template, new_global)
        final_reward = float((final_accuracy - baseline_accuracy) / 100.0)
        self.last_validation_accuracy = final_accuracy

        if rollout_records:
            best_rollout = max(
                rollout_records, key=lambda item: item["validation_accuracy"]
            )
            rollout_rewards = [item["reward"] for item in rollout_records]
        else:
            best_rollout = None
            rollout_rewards = []

        meta = {
            "round": int(round_idx),
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
            "rollouts_per_round": int(self.rollouts_per_round),
            "num_rollouts_executed": int(len(rollout_records)),
            "num_transitions_added": int(len(action_cache)),
            "rollout_reward_mean": float(np.mean(rollout_rewards))
            if rollout_rewards
            else 0.0,
            "rollout_reward_max": float(np.max(rollout_rewards))
            if rollout_rewards
            else 0.0,
            "best_rollout": best_rollout,
            "rollouts": rollout_records,
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
    rule_rl_server = RuleRLServer(args, dataset_test, central_dataset)

    base_info = get_base_info(args)
    filename = "./" + args.save + "/accuracy_file_{}.txt".format(base_info)
    wandb_run = init_wandb(args, base_info, len(central_dataset), len(eval_indices))

    val_acc_list = [0.0001]
    backdoor_acculist = [0]
    loss_train = []
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

        w_glob, rule_rl_meta = rule_rl_server.aggregate(
            iter_idx, net_glob, round_w_locals, round_w_updates
        )
        action = rule_rl_meta["action"]
        print(
            "RuleBandit[{}] round {:3d}: unique/exec {}/{} | prefilter {}/{} | selected {} | "
            "commit action {}({},{}) | val_acc {:.2f} | reward {:.4f} | "
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
            wandb_run.log(
                build_wandb_metrics(
                    iter_idx, loss_avg, main_acc, backdoor_acc, rule_rl_meta, round_attack_count
                ),
                step=iter_idx,
            )
        write_file(filename, val_acc_list, backdoor_acculist, args)

    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)
    torch.save(rule_rl_server.history, "./" + args.save + "/rule_rl_history.pt")

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
