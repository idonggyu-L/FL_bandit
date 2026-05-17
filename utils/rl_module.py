# dqn_rule_agent_test_donut.py

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Dict, Any, Optional

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# 1. Action space (도넛 형태 필터링 추가)
# ============================================================

@dataclass(frozen=True)
class RuleAction:
    """
    Bandit/RL action은 client id가 아니라 selection rule 하나를 고르는 것.
    이제 도넛 형태(Band-pass) 필터링을 지원합니다.

    예:
        drop_inner_ratio = 0.0, keep_outer_ratio = 0.5 -> 중심부터 상위 50% 선택 (꽉 찬 원)
        drop_inner_ratio = 0.1, keep_outer_ratio = 0.8 -> 최상위 10%는 버리고, 10%~80% 구간만 선택 (도넛)
    """
    score_type: str
    center_type: str
    drop_inner_ratio: float  # 중심부에서 버릴 비율
    keep_outer_ratio: float  # 외곽 한계선 비율
    knn_k: int = 3


def build_rule_actions(
    # (drop_inner, keep_outer) 조합
    ratio_pairs: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.5),  
        (0.0, 0.8),  
        (0.1, 0.8),  # 도넛: 최상위 10% 제외, 10~80% 구간 (너무 뻔한 값 버림)
        (0.2, 0.9),  # 얇은 도넛: 최상위 20% 제외, 20~90% 구간
        (0.0, 1.0)   # 전체 선택 (필터링 없음)
    ),
    score_types: Tuple[str, ...] = ("l2_to_center", "cosine_to_center", "knn_distance"),
    center_types: Tuple[str, ...] = ("mean", "median", "medoid"),
    knn_k: int = 3,
) -> List[RuleAction]:
    actions: List[RuleAction] = []

    for score_type in score_types:
        if score_type == "knn_distance":
            for (drop_inner, keep_outer) in ratio_pairs:
                actions.append(
                    RuleAction(
                        score_type=score_type,
                        center_type="none",
                        drop_inner_ratio=drop_inner,
                        keep_outer_ratio=keep_outer,
                        knn_k=knn_k,
                    )
                )
        else:
            for center_type in center_types:
                for (drop_inner, keep_outer) in ratio_pairs:
                    actions.append(
                        RuleAction(
                            score_type=score_type,
                            center_type=center_type,
                            drop_inner_ratio=drop_inner,
                            keep_outer_ratio=keep_outer,
                            knn_k=knn_k,
                        )
                    )

    return actions


# ============================================================
# 2. Replay buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch and return tensors.

        Args:
            batch_size: number of samples to draw
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # allow callers to pass either a torch.device or None
        # default to CPU when device is None
        # using torch.tensor here is fine for small batches
        states = torch.tensor(np.array(states), dtype=torch.float32, device=None)
        actions = torch.tensor(actions, dtype=torch.long, device=None).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=None).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=None)
        dones = torch.tensor(dones, dtype=torch.float32, device=None).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class BanditReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
    ):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
            )
        )

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 3. Q-network
# ============================================================

class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralLinearNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 128,
        feature_dim: int = 64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(feature_dim, num_actions)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


BANDIT_MODES = ("LinUCB", "NeuralUCB", "NeuralTS", "NeuralLinear")


def normalize_bandit_mode(mode: str) -> str:
    key = mode.replace("_", "").replace("-", "").lower()
    mapping = {
        "linucb": "LinUCB",
        "neuralucb": "NeuralUCB",
        "neuralts": "NeuralTS",
        "neurallinear": "NeuralLinear",
    }
    if key not in mapping:
        raise ValueError(
            f"Unknown bandit mode '{mode}'. Expected one of: {', '.join(BANDIT_MODES)}"
        )
    return mapping[key]


@dataclass
class BanditConfig:
    state_dim: int
    num_actions: int

    mode: str = "LinUCB"
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 20_000

    lambda_reg: float = 1.0
    alpha: float = 1.0
    nu: float = 1.0

    hidden_dim: int = 128
    feature_dim: int = 64
    train_steps_per_update: int = 1

    grad_clip_norm: float = 5.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BaseRuleBanditAgent:
    def __init__(
        self,
        actions: List[RuleAction],
        config: BanditConfig,
    ):
        self.actions = actions
        self.config = config
        self.mode = normalize_bandit_mode(config.mode)
        self.device = torch.device(config.device)

        if config.num_actions != len(actions):
            raise ValueError("config.num_actions must match len(actions)")

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.replay_buffer = BanditReplayBuffer(config.replay_capacity)
        self.pending_transitions: List[Tuple[np.ndarray, int, float]] = []
        self.train_steps = 0
        self.epsilon = 0.0

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray] = None,
        done: bool = True,
    ):
        del next_state, done
        transition = (np.asarray(state, dtype=np.float32), int(action), float(reward))
        self.replay_buffer.push(*transition)
        self.pending_transitions.append(transition)

    def _drain_pending(self) -> List[Tuple[np.ndarray, int, float]]:
        pending = self.pending_transitions
        self.pending_transitions = []
        return pending

    def get_action_object(self, action_id: int) -> RuleAction:
        if action_id < 0 or action_id >= len(self.actions):
            raise IndexError(f"action_id {action_id} out of range [0, {len(self.actions)-1}]")
        return self.actions[action_id]

    @property
    def exploration_value(self) -> float:
        if self.mode in ("LinUCB", "NeuralUCB"):
            return float(self.config.alpha)
        return float(self.config.nu)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        raise NotImplementedError

    def update(self) -> Optional[Dict[str, float]]:
        raise NotImplementedError

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinUCBRuleAgent(BaseRuleBanditAgent):
    def __init__(
        self,
        actions: List[RuleAction],
        config: BanditConfig,
    ):
        super().__init__(actions, config)
        self.A = np.stack(
            [
                config.lambda_reg * np.eye(config.state_dim, dtype=np.float64)
                for _ in range(config.num_actions)
            ],
            axis=0,
        )
        self.b = np.zeros((config.num_actions, config.state_dim), dtype=np.float64)
        self.action_counts = np.zeros(config.num_actions, dtype=np.int64)

    def _posterior(self, action_id: int) -> Tuple[np.ndarray, np.ndarray]:
        a_inv = np.linalg.inv(self.A[action_id])
        theta = a_inv @ self.b[action_id]
        return theta, a_inv

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        x = np.asarray(state, dtype=np.float64)
        scores = []
        for action_id in range(self.config.num_actions):
            theta, a_inv = self._posterior(action_id)
            mean = float(x @ theta)
            bonus = 0.0
            if not eval_mode:
                bonus = self.config.alpha * float(np.sqrt(max(x @ a_inv @ x, 0.0)))
            scores.append(mean + bonus)
        return int(np.argmax(scores))

    def update(self) -> Optional[Dict[str, float]]:
        pending = self._drain_pending()
        if not pending:
            return None

        rewards = []
        for state, action_id, reward in pending:
            x = np.asarray(state, dtype=np.float64)
            self.A[action_id] += np.outer(x, x)
            self.b[action_id] += reward * x
            self.action_counts[action_id] += 1
            rewards.append(reward)

        return {
            "num_updates": float(len(pending)),
            "reward_mean": float(np.mean(rewards)),
            "exploration": self.exploration_value,
        }

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64)
        values = []
        for action_id in range(self.config.num_actions):
            theta, _ = self._posterior(action_id)
            values.append(float(x @ theta))
        return np.asarray(values, dtype=np.float32)


class NeuralUncertaintyRuleAgent(BaseRuleBanditAgent):
    def __init__(
        self,
        actions: List[RuleAction],
        config: BanditConfig,
        mode: str,
    ):
        super().__init__(actions, config)
        self.mode = mode
        self.reward_net = QNetwork(
            state_dim=config.state_dim,
            num_actions=config.num_actions,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=config.lr)
        self.precision = torch.full(
            (self._num_parameters(),),
            float(config.lambda_reg),
            dtype=torch.float32,
            device=self.device,
        )

    def _num_parameters(self) -> int:
        return sum(param.numel() for param in self.reward_net.parameters())

    def _action_grad(self, state: np.ndarray, action_id: int) -> torch.Tensor:
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        value = self.reward_net(state_tensor)[0, action_id]
        grads = torch.autograd.grad(
            value,
            tuple(self.reward_net.parameters()),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        return torch.cat([grad.detach().reshape(-1) for grad in grads])

    def _uncertainty(self, state: np.ndarray, action_id: int) -> float:
        grad = self._action_grad(state, action_id)
        var = torch.sum((grad * grad) / torch.clamp(self.precision, min=1e-8))
        return float(torch.sqrt(torch.clamp(var, min=0.0)).item())

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            means = self.reward_net(state_tensor).squeeze(0).detach().cpu().numpy()

        if eval_mode:
            return int(np.argmax(means))

        scores = []
        for action_id, mean in enumerate(means):
            uncertainty = self._uncertainty(state, action_id)
            if self.mode == "NeuralUCB":
                score = float(mean) + self.config.alpha * uncertainty
            else:
                score = float(mean) + np.random.normal(
                    loc=0.0,
                    scale=self.config.nu * uncertainty,
                )
            scores.append(score)
        return int(np.argmax(scores))

    def _train_reward_net(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        losses = []
        for _ in range(max(1, self.config.train_steps_per_update)):
            states, actions, rewards = self.replay_buffer.sample(
                self.config.batch_size, self.device
            )
            pred = self.reward_net(states).gather(1, actions)
            loss = F.smooth_l1_loss(pred, rewards)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.reward_net.parameters(), self.config.grad_clip_norm
            )
            self.optimizer.step()
            losses.append(float(loss.item()))
            self.train_steps += 1

        return float(np.mean(losses))

    def update(self) -> Optional[Dict[str, float]]:
        pending = self._drain_pending()
        loss = self._train_reward_net()

        for state, action_id, _ in pending:
            grad = self._action_grad(state, action_id)
            self.precision += grad * grad

        if loss is None and not pending:
            return None

        return {
            "loss": float(loss) if loss is not None else 0.0,
            "num_updates": float(len(pending)),
            "precision_mean": float(self.precision.mean().item()),
            "exploration": self.exploration_value,
        }

    @torch.no_grad()
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return self.reward_net(state_tensor).squeeze(0).detach().cpu().numpy()


class NeuralUCBRuleAgent(NeuralUncertaintyRuleAgent):
    def __init__(
        self,
        actions: List[RuleAction],
        config: BanditConfig,
    ):
        super().__init__(actions, config, mode="NeuralUCB")


class NeuralTSRuleAgent(NeuralUncertaintyRuleAgent):
    def __init__(
        self,
        actions: List[RuleAction],
        config: BanditConfig,
    ):
        super().__init__(actions, config, mode="NeuralTS")


class NeuralLinearRuleAgent(BaseRuleBanditAgent):
    def __init__(
        self,
        actions: List[RuleAction],
        config: BanditConfig,
    ):
        super().__init__(actions, config)
        self.model = NeuralLinearNetwork(
            state_dim=config.state_dim,
            num_actions=config.num_actions,
            hidden_dim=config.hidden_dim,
            feature_dim=config.feature_dim,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.A = np.stack(
            [
                config.lambda_reg * np.eye(config.feature_dim, dtype=np.float64)
                for _ in range(config.num_actions)
            ],
            axis=0,
        )
        self.b = np.zeros((config.num_actions, config.feature_dim), dtype=np.float64)

    @torch.no_grad()
    def _features_np(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        features = self.model.features(state_tensor).squeeze(0)
        return features.detach().cpu().numpy().astype(np.float64)

    def _posterior(self, action_id: int) -> Tuple[np.ndarray, np.ndarray]:
        a_inv = np.linalg.inv(self.A[action_id])
        theta = a_inv @ self.b[action_id]
        return theta, a_inv

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        phi = self._features_np(state)
        scores = []
        for action_id in range(self.config.num_actions):
            theta, a_inv = self._posterior(action_id)
            mean = float(phi @ theta)
            if eval_mode:
                score = mean
            else:
                std = float(np.sqrt(max(phi @ a_inv @ phi, 0.0)))
                score = mean + np.random.normal(loc=0.0, scale=self.config.nu * std)
            scores.append(score)
        return int(np.argmax(scores))

    def _train_encoder(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        losses = []
        for _ in range(max(1, self.config.train_steps_per_update)):
            states, actions, rewards = self.replay_buffer.sample(
                self.config.batch_size, self.device
            )
            pred = self.model(states).gather(1, actions)
            loss = F.smooth_l1_loss(pred, rewards)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.optimizer.step()
            losses.append(float(loss.item()))
            self.train_steps += 1

        return float(np.mean(losses))

    def update(self) -> Optional[Dict[str, float]]:
        pending = self._drain_pending()
        loss = self._train_encoder()

        rewards = []
        for state, action_id, reward in pending:
            phi = self._features_np(state)
            self.A[action_id] += np.outer(phi, phi)
            self.b[action_id] += reward * phi
            rewards.append(reward)

        if loss is None and not pending:
            return None

        return {
            "loss": float(loss) if loss is not None else 0.0,
            "num_updates": float(len(pending)),
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "exploration": self.exploration_value,
        }

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        phi = self._features_np(state)
        values = []
        for action_id in range(self.config.num_actions):
            theta, _ = self._posterior(action_id)
            values.append(float(phi @ theta))
        return np.asarray(values, dtype=np.float32)


def create_rule_bandit_agent(
    actions: List[RuleAction],
    config: BanditConfig,
) -> BaseRuleBanditAgent:
    mode = normalize_bandit_mode(config.mode)
    config.mode = mode
    if mode == "LinUCB":
        return LinUCBRuleAgent(actions, config)
    if mode == "NeuralUCB":
        return NeuralUCBRuleAgent(actions, config)
    if mode == "NeuralTS":
        return NeuralTSRuleAgent(actions, config)
    if mode == "NeuralLinear":
        return NeuralLinearRuleAgent(actions, config)
    raise AssertionError(f"unreachable bandit mode: {mode}")


# ============================================================
# 4. DQN config
# ============================================================

@dataclass
class DQNConfig:
    state_dim: int
    num_actions: int

    gamma: float = 0.0
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 20_000

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    target_update_interval: int = 100
    hidden_dim: int = 128

    grad_clip_norm: float = 5.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 5. DQN agent
# ============================================================

class DQNRuleAgent:
    def __init__(
        self,
        actions: List[RuleAction],
        config: DQNConfig,
    ):
        self.actions = actions
        self.config = config

        assert config.num_actions == len(actions)

        self.device = torch.device(config.device)

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # if CUDA available and requested, also seed CUDA and set deterministic flags
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            # keep deterministic behavior for reproducibility where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.q_net = QNetwork(
            state_dim=config.state_dim,
            num_actions=config.num_actions,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim=config.state_dim,
            num_actions=config.num_actions,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.replay_capacity)

        self.epsilon = config.epsilon_start
        self.train_steps = 0

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        eval_mode: bool = False,
    ) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.config.num_actions)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        q_values = self.q_net(state_tensor)
        action_id = int(torch.argmax(q_values, dim=1).item())
        return action_id

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_values.mean().item()),
            "target_q_mean": float(target_q.mean().item()),
            "epsilon": float(self.epsilon),
        }

    @torch.no_grad()
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        q_values = self.q_net(state_tensor).squeeze(0)
        return q_values.detach().cpu().numpy()

    def get_action_object(self, action_id: int) -> RuleAction:
        if action_id < 0 or action_id >= len(self.actions):
            raise IndexError(f"action_id {action_id} out of range [0, {len(self.actions)-1}]")
        return self.actions[action_id]


# ============================================================
# 6. Dummy environment for testing
# ============================================================

class DummyRepresentationRuleEnv:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        rollouts_per_round: int = 100,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.rollouts_per_round = rollouts_per_round

        self.rng = np.random.default_rng(seed)

        self.hidden_w = self.rng.normal(
            loc=0.0,
            scale=0.7,
            size=(num_actions, state_dim),
        )
        self.hidden_b = self.rng.normal(
            loc=0.0,
            scale=0.2,
            size=(num_actions,),
        )

        self.state: Optional[np.ndarray] = None
        self.t = 0

    def reset(self) -> np.ndarray:
        self.state = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(self.state_dim,),
        ).astype(np.float32)

        self.t = 0
        return self.state.copy()

    def _oracle_reward(self, state: np.ndarray, action_id: int) -> float:
        logit = float(state @ self.hidden_w[action_id] + self.hidden_b[action_id])
        reward = 1.0 / (1.0 + np.exp(-logit))

        noise = self.rng.normal(loc=0.0, scale=0.03)
        reward = reward + noise

        return float(np.clip(reward, 0.0, 1.0))

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self.state is not None
        reward = self._oracle_reward(self.state, action_id)
        self.t += 1
        done = self.t >= self.rollouts_per_round
        next_state = self.state.copy()

        info = {
            "rollout_idx": self.t,
            "action_id": action_id,
            "reward": reward,
        }
        return next_state, reward, done, info

    def oracle_best_action(self, state: np.ndarray) -> int:
        scores = []
        for action_id in range(self.num_actions):
            logit = float(state @ self.hidden_w[action_id] + self.hidden_b[action_id])
            reward = 1.0 / (1.0 + np.exp(-logit))
            scores.append(reward)
        return int(np.argmax(scores))


# ============================================================
# 7. Dummy train test
# ============================================================

def run_dummy_dqn_test():
    state_dim = 14

    # 도넛 형태를 포함한 새로운 액션 공간 빌드
    actions = build_rule_actions(
        ratio_pairs=(
            (0.0, 0.5), (0.0, 0.8), # 일반 원
            (0.1, 0.8), (0.2, 0.9), # 도넛(대역통과)
            (0.0, 1.0)              # 전체
        ),
        score_types=("l2_to_center", "cosine_to_center", "knn_distance"),
        center_types=("mean", "median", "medoid"),
        knn_k=3,
    )

    num_actions = len(actions)
    print(f"Total number of actions: {num_actions}")

    config = DQNConfig(
        state_dim=state_dim,
        num_actions=num_actions,
        gamma=0.0,
        lr=1e-3,
        batch_size=64,
        replay_capacity=20_000,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
        target_update_interval=100,
        hidden_dim=128,
        seed=42,
    )

    agent = DQNRuleAgent(actions=actions, config=config)
    env = DummyRepresentationRuleEnv(
        state_dim=state_dim,
        num_actions=num_actions,
        rollouts_per_round=100,
        seed=123,
    )

    num_rounds = 200
    recent_rewards = deque(maxlen=1000)
    recent_oracle_match = deque(maxlen=1000)

    for round_idx in range(num_rounds):
        state = env.reset()

        for rollout_idx in range(env.rollouts_per_round):
            action_id = agent.select_action(state)
            next_state, reward, done, info = env.step(action_id)

            agent.push_transition(state, action_id, reward, next_state, done)
            agent.update()

            oracle_action = env.oracle_best_action(state)
            recent_oracle_match.append(float(action_id == oracle_action))
            recent_rewards.append(reward)

            state = next_state
            if done:
                break

        if (round_idx + 1) % 20 == 0:
            avg_reward = float(np.mean(recent_rewards))
            match_rate = float(np.mean(recent_oracle_match))
            print(
                f"[round {round_idx + 1:03d}] "
                f"avg_reward={avg_reward:.4f} | "
                f"oracle_match={match_rate:.4f} | "
                f"epsilon={agent.epsilon:.4f}"
            )

    test_state = env.reset()
    action_id = agent.select_action(test_state, eval_mode=True)
    action = agent.get_action_object(action_id)

    print("\n=== Final test ===")
    print("DQN selected action id:", action_id)
    print(f"DQN selected action: {action.score_type}, {action.center_type}")
    print(f"  -> Inner drop: {action.drop_inner_ratio*100}%, Outer keep: {action.keep_outer_ratio*100}%")

if __name__ == "__main__":
    run_dummy_dqn_test()

