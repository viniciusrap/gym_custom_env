"""
train_grid_world_cpp.py
-----------------------
Train, evaluate and visualize a PPO agent on the Coverage Path Planning
(CPP) environment, with curriculum learning and transfer between grid sizes.

Command-line interface (matches the assignment spec)
----------------------------------------------------
    python train_grid_world_cpp.py train      <size> <obstacles> <max_steps> <timesteps>
    python train_grid_world_cpp.py test       <size> <obstacles>
    python train_grid_world_cpp.py run        <size> <obstacles>
    python train_grid_world_cpp.py curriculum <size> <obstacles> <max_steps> <timesteps>

Examples
--------
    python train_grid_world_cpp.py train      5 3 200 500000
    python train_grid_world_cpp.py test       5 3
    python train_grid_world_cpp.py run        5 3
    python train_grid_world_cpp.py curriculum 5 3 200 500000

Folder layout
-------------
    models/                              final approved checkpoints
        cpp_5x5_approved.zip
        cpp_10x10_approved.zip
        cpp_20x20_approved.zip
    checkpoints/                         intermediate snapshots
        cpp_5x5_step_<num>.zip
        ...
    log/                                 TensorBoard logs (one folder per stage)
    results/                             evaluation CSVs

Approval rule
-------------
A model is "approved" for a given grid size when its mean coverage over
EVAL_EPISODES (=100) evaluation episodes is > APPROVAL_THRESHOLD (=0.90).
Previously approved models are NEVER overwritten silently — they are moved to
a timestamped backup before being replaced. Failed stages do not destroy
earlier approved models, so you can adjust hyperparameters and resume from
the failing stage.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecFrameStack,
)
from sb3_contrib import RecurrentPPO  # PPO + LSTM for POMDP / partial observability
import torch  # for AdamW

import gymnasium_env  # noqa: F401  (registers GridWorldCPP-v0)
from cpp_policy import CPPFeatureExtractor


# Force stdout to be line-buffered & unbuffered so prints appear immediately
# even when piped or running in PowerShell. Without this, you can see long
# silent gaps that look like the script is frozen when it's actually running.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    # Older Python or non-TTY stdout — fall back silently.
    pass


def _ts() -> str:
    """Short timestamp prefix for heartbeat prints."""
    return datetime.now().strftime("%H:%M:%S")


def hb(msg: str) -> None:
    """Heartbeat print: timestamped, immediately flushed.

    Use this for any phase boundary so the user knows the script is alive
    even when there's no progress bar / tqdm output for a few seconds."""
    print(f"[{_ts()}] {msg}", flush=True)


class HeartbeatCallback(BaseCallback):
    """Prints a 'still alive' line every N seconds with current progress.

    The SB3 progress bar (tqdm) updates in-place using \\r, which means if
    the process freezes the bar simply stops moving — there's no separate
    line saying 'still going'. This callback prints a fresh, timestamped
    line at a fixed wall-clock interval. If those lines stop appearing,
    the process is genuinely stuck.
    """

    def __init__(self, every_seconds: int = 30, total_timesteps: int | None = None):
        super().__init__()
        self.every_seconds = every_seconds
        self._last_print = time.monotonic()
        self._start = time.monotonic()
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        now = time.monotonic()
        if now - self._last_print >= self.every_seconds:
            elapsed = int(now - self._start)
            steps = self.num_timesteps
            if self.total_timesteps:
                pct = 100.0 * steps / max(self.total_timesteps, 1)
                # Linear ETA based on elapsed/steps so far.
                if steps > 0:
                    eta = int(elapsed * (self.total_timesteps - steps) / steps)
                    eta_str = f"{eta // 60}m{eta % 60:02d}s"
                else:
                    eta_str = "?"
                hb(f"  [training alive] {steps:>9,}/{self.total_timesteps:,} "
                   f"steps ({pct:5.1f}%), elapsed {elapsed // 60}m{elapsed % 60:02d}s, "
                   f"ETA ~{eta_str}")
            else:
                hb(f"  [training alive] {steps:,} steps, "
                   f"elapsed {elapsed // 60}m{elapsed % 60:02d}s")
            self._last_print = now
        return True


# --------------------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "log"
RESULTS_DIR = ROOT / "results"
for d in (MODELS_DIR, CHECKPOINTS_DIR, LOG_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- config

# How many parallel rollout workers — auto-detect, capped at 16. Override via
# CPP_N_ENVS env variable, e.g.  set CPP_N_ENVS=16  on the desktop machine.
def _default_n_envs() -> int:
    cpu = os.cpu_count() or 4
    return max(2, min(cpu, 16))


N_ENVS = int(os.environ.get("CPP_N_ENVS", _default_n_envs()))

# Frame stacking: with RecurrentPPO + LSTM, we DISABLE frame stacking because
# the LSTM hidden state already provides unlimited memory of past observations.
# Stacking on top would just duplicate information at the input level. Setting
# to 1 effectively disables it (the env returns its native (3,3) obs).
FRAME_STACK = 1

# Approval threshold: the assignment requires coverage CLOSE TO 100%.
# 0.90 is the minimum "good enough" bar — anything below means failure.
# We aim for >= 0.95 in practice; the rubric rewards proximity to 100%.
APPROVAL_THRESHOLD = 0.90

# Number of evaluation episodes used for approval / reporting.
# Raised from 100 to 200 after observing high std (±25%) in some runs:
# 100 episodes left a CI half-width of ~5% which is too wide to distinguish
# 88% from 92%. 200 episodes cuts that roughly to ~3.5%.
EVAL_EPISODES = 200

# Curriculum scaling: stage[i+1] uses (size*sf, obs*of, steps*mf, timesteps*tf)
# applied to the values provided on the command line for stage 1.
#
# With the prompt's defaults (5 / 3 / 200 / 500_000) you get:
#   Stage 1:  5x5,  3 obstacles,  200 steps,    500_000 timesteps   (from scratch)
#   Stage 2: 10x10, 12 obstacles,  500 steps,  1_500_000 timesteps  (transfer)
#   Stage 3: 15x15, 27 obstacles, 1000 steps,  3_000_000 timesteps  (transfer) [INTERMEDIATE]
#   Stage 4: 20x20, 48 obstacles, 1500 steps,  5_000_000 timesteps  (transfer)
#
# Rationale: jumping directly from 10x10 (88 free cells) to 20x20 (352 free cells)
# is a 4x increase in search space and the agent struggles to generalize the
# coverage strategy at that scale. We added an intermediate 15x15 stage as a
# softer step (10x10 -> 15x15: 2.6x scale; 15x15 -> 20x20: 1.6x scale) which
# is a standard curriculum learning technique for hard generalization problems.
CURRICULUM_SCALING = [
    # stage 2: 10x10, 12 obs, 500 steps, 1.5M timesteps
    (2, 4, 2.5, 3),
    # stage 3: 15x15, 27 obs, 1000 steps, 3M timesteps  [NEW intermediate stage]
    #   size*3 = 15, obs*9 = 27, steps*5 = 1000, timesteps*6 = 3_000_000
    (3, 9, 5, 6),
    # stage 4: 20x20, 48 obs, 1500 steps, 5M timesteps
    (4, 16, 7.5, 10),
]


@dataclass
class StageResult:
    name: str
    size: int
    obstacles: int
    timesteps: int
    mean_coverage: float
    std_coverage: float
    full_coverage_rate: float
    mean_reward: float
    std_reward: float
    mean_steps: float
    best_episode_idx: int
    best_episode_coverage: float
    best_episode_reward: float
    best_episode_steps: int
    approved: bool
    model_path: str


# --------------------------------------------------------------------------- envs

def _env_factory(size: int, obstacles: int, max_steps: int,
                 render_mode: str | None = None):
    """Factory closure for SubprocVecEnv. Each worker calls this; we (re)import
    gymnasium_env here so the env id is registered inside that process."""
    def _thunk():
        import gymnasium_env  # noqa: F401
        return gym.make(
            "GridWorldCPP-v0",
            size=size, obs_quantity=obstacles, max_steps=max_steps,
            render_mode=render_mode,
        )
    return _thunk


def make_env(size: int, obstacles: int, max_steps: int, n_envs: int = N_ENVS,
             stack: bool = True):
    """Vectorized (+ frame-stacked) env for training."""
    factory = _env_factory(size, obstacles, max_steps)
    if n_envs == 1:
        vec = make_vec_env(factory, n_envs=1, vec_env_cls=DummyVecEnv)
    else:
        vec = make_vec_env(factory, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    if stack and FRAME_STACK > 1:
        vec = VecFrameStack(vec, n_stack=FRAME_STACK)
    return vec


def make_single_env(size: int, obstacles: int, max_steps: int,
                    render_mode: str | None = None,
                    stack: bool = True):
    """Single (non-parallel) env for evaluation / visualization."""
    factory = _env_factory(size, obstacles, max_steps, render_mode=render_mode)
    vec = make_vec_env(factory, n_envs=1, vec_env_cls=DummyVecEnv)
    if stack and FRAME_STACK > 1:
        vec = VecFrameStack(vec, n_stack=FRAME_STACK)
    return vec


# --------------------------------------------------------------------------- ppo

def make_ppo(env, tensorboard_log: str | None = None) -> RecurrentPPO:
    """Build a fresh RecurrentPPO agent with custom feature extractor + LSTM.

    Why RecurrentPPO instead of plain PPO:
    -------------------------------------
    Earlier runs with frame-stacked PPO (n_stack=8) plateaued at ~86% on
    10x10 with std ~22% — bimodal performance indicating perceptual aliasing
    (different global states map to identical 3x3 windows + recent history).
    The literature on POMDP with limited observability is unanimous: LSTM-
    based policies dominate frame stacking.
      - "Learning Robust Policies under Partial Observability" (arXiv 2509.20008,
        Sept 2025): RNN converged 3x faster than frame stacking on partial-obs
        cybersecurity benchmark.
      - SB3 contrib RecurrentPPO is the standard PyTorch implementation,
        same training algorithm as PPO with LSTM hidden state propagation.

    Optimizer is AdamW with weight_decay=1e-4 (Dohare et al. Nature 2024 —
    keeps plasticity across curriculum stages).
    """
    policy_kwargs = dict(
        features_extractor_class=CPPFeatureExtractor,
        features_extractor_kwargs=dict(cnn_features=64, scalar_features=32),
        net_arch=dict(pi=[64], vf=[64]),  # smaller MLP heads, LSTM does heavy lifting
        # LSTM config:
        lstm_hidden_size=128,
        n_lstm_layers=1,
        shared_lstm=False,        # separate LSTM for actor and critic — more stable
        enable_critic_lstm=True,  # critic also gets memory
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-4),
    )
    return RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=env,
        verbose=1,
        device="cpu",
        learning_rate=3e-4,
        n_steps=128,            # smaller rollout per env: LSTM truncated BPTT works
                                # better with shorter sequences; 128 * N_ENVS = enough
                                # samples per update
        batch_size=128,         # must divide (n_steps * n_envs) for sequence batching
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
    )


def load_ppo_for_transfer(model_path: Path, env,
                          tensorboard_log: str | None = None) -> RecurrentPPO:
    """Load a saved RecurrentPPO and attach to a (different-size) env."""
    custom_objects = {"learning_rate": 1e-4, "ent_coef": 0.005}
    return RecurrentPPO.load(
        str(model_path), env=env, device="cpu",
        tensorboard_log=tensorboard_log, custom_objects=custom_objects,
    )


# --------------------------------------------------------------------------- eval

def evaluate(model: PPO, size: int, obstacles: int, max_steps: int,
             n_episodes: int = EVAL_EPISODES, deterministic: bool = True,
             save_csv: Path | None = None,
             progress_every: int = 10) -> dict:
    """Roll out n_episodes and return summary metrics including the best episode.

    Prints a heartbeat every `progress_every` episodes so the user can see
    progress during what would otherwise be a long silent block (especially
    on 20x20 with 1500 max_steps, where 100 episodes can take ~3 minutes)."""
    hb(f"  [evaluate] starting {n_episodes} episodes "
       f"on {size}x{size} ({obstacles} obstacles, max_steps={max_steps})")
    eval_start = time.monotonic()
    env = make_single_env(size, obstacles, max_steps, render_mode=None, stack=True)

    coverages: list[float] = []
    steps_list: list[int] = []
    rewards_list: list[float] = []
    full_count = 0

    # Hard safety cap on per-episode length: even if the env breaks and never
    # returns done=True, we abort after this many predict() calls. Set to
    # 2x the env's max_steps (the env should never let an episode go beyond
    # max_steps, but we don't trust unconditionally).
    HARD_STEP_CAP = max_steps * 2 + 10

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        last_info: dict = {}
        step_in_episode = 0
        # LSTM hidden state: must be reset at the start of each episode and
        # propagated across predict() calls within the episode.
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)  # signals "new episode"
        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            obs, reward, dones, infos = env.step(action)
            ep_reward += float(reward[0])
            last_info = infos[0]
            done = bool(dones[0])
            episode_starts = np.zeros((1,), dtype=bool)  # not a new episode anymore
            step_in_episode += 1
            if step_in_episode >= HARD_STEP_CAP:
                hb(f"  [evaluate] WARNING: episode {ep} hit hard step cap "
                   f"({HARD_STEP_CAP}); env may be misbehaving. Aborting episode.")
                break

        cov = float(last_info.get("coverage_ratio", 0.0))
        coverages.append(cov)
        steps_list.append(int(last_info.get("steps", 0)))
        rewards_list.append(ep_reward)
        if cov >= 0.999999:
            full_count += 1

        # Periodic progress print so long evaluations don't look frozen.
        if (ep + 1) % progress_every == 0 or (ep + 1) == n_episodes:
            elapsed = time.monotonic() - eval_start
            running_mean = float(np.mean(coverages))
            hb(f"  [evaluate] {ep + 1:>3}/{n_episodes} episodes done "
               f"({elapsed:>5.1f}s, running mean coverage = {running_mean:.2%})")

    env.close()

    coverages_np = np.array(coverages)
    rewards_np = np.array(rewards_list)
    steps_np = np.array(steps_list)

    # Best episode = highest coverage (ties broken by reward)
    sort_key = coverages_np * 1e6 + rewards_np
    best_idx = int(np.argmax(sort_key))

    summary = dict(
        mean_coverage=float(coverages_np.mean()),
        std_coverage=float(coverages_np.std()),
        max_coverage=float(coverages_np.max()),
        min_coverage=float(coverages_np.min()),
        full_coverage_rate=full_count / n_episodes,
        mean_reward=float(rewards_np.mean()),
        std_reward=float(rewards_np.std()),
        mean_steps=float(steps_np.mean()),
        std_steps=float(steps_np.std()),
        best_episode_idx=best_idx,
        best_episode_coverage=float(coverages_np[best_idx]),
        best_episode_reward=float(rewards_np[best_idx]),
        best_episode_steps=int(steps_np[best_idx]),
    )

    if save_csv is not None:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "coverage", "steps", "reward"])
            for i, (c, s, r) in enumerate(zip(coverages, steps_list, rewards_list)):
                w.writerow([i, c, s, r])

    return summary


def print_eval_summary(name: str, size: int, obstacles: int, summary: dict,
                       model_path: str | None = None) -> None:
    print()
    print("=" * 64)
    print(f" Evaluation: {name}  (grid {size}x{size}, {obstacles} obstacles)")
    print("=" * 64)
    print(f"  Mean coverage         : {summary['mean_coverage']:.2%} "
          f"+/- {summary['std_coverage']:.2%}")
    print(f"  Min / Max coverage    : "
          f"{summary['min_coverage']:.2%} / {summary['max_coverage']:.2%}")
    print(f"  Full coverage rate    : {summary['full_coverage_rate']:.1%} "
          f"(of {EVAL_EPISODES} episodes)")
    print(f"  Mean reward           : {summary['mean_reward']:.2f} "
          f"+/- {summary['std_reward']:.2f}")
    print(f"  Mean steps            : {summary['mean_steps']:.1f} "
          f"+/- {summary['std_steps']:.1f}")
    print(f"  Best episode (#{summary['best_episode_idx']})  : "
          f"coverage {summary['best_episode_coverage']:.2%}, "
          f"reward {summary['best_episode_reward']:.2f}, "
          f"steps {summary['best_episode_steps']}")
    if model_path:
        print(f"  Model                 : {model_path}")
    print("=" * 64)


# --------------------------------------------------------------------------- approval

def _backup_if_exists(path: Path) -> None:
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.bak_{ts}{path.suffix}")
        path.rename(backup)
        print(f"  [backup] previous {path.name} moved to {backup.name}")


def _save_if_approved(model: PPO, size: int, summary: dict) -> tuple[bool, str]:
    approved = summary["mean_coverage"] > APPROVAL_THRESHOLD
    approved_path = MODELS_DIR / f"cpp_{size}x{size}_approved.zip"
    if approved:
        _backup_if_exists(approved_path)
        model.save(str(approved_path))
        print(f"  [APPROVED] saved as {approved_path}")
        return True, str(approved_path)
    else:
        print(f"  [NOT APPROVED] mean coverage {summary['mean_coverage']:.2%} "
              f"<= {APPROVAL_THRESHOLD:.0%}")
        return False, ""


# --------------------------------------------------------------------------- modes

def cmd_train(size: int, obstacles: int, max_steps: int, timesteps: int,
              source_model: str | None = None,
              save_name: str | None = None,
              register_as_official: bool = True) -> StageResult:
    """Train (from scratch or via transfer) on a single grid configuration.

    Args:
        register_as_official: when True (default), if the model passes the
            approval threshold it is also saved under the canonical name
            cpp_<size>x<size>_approved.zip (with backup of any prior). Set
            to False when running an intermediate sub-stage of a sub-curriculum
            (e.g. bigtwenty's "easy" 20x20 with only 16 obstacles) that should
            NOT compete for the official slot — only the final sub-stage with
            the assignment-level configuration should register as official.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = save_name or f"cpp_{size}x{size}_{timestamp}"
    tb_log = str(LOG_DIR / name)

    hb(f"[cmd_train] building {N_ENVS} parallel envs ({size}x{size}, "
       f"{obstacles} obs, max_steps={max_steps})")
    env_build_start = time.monotonic()
    env = make_env(size, obstacles, max_steps, n_envs=N_ENVS)
    hb(f"[cmd_train] envs ready in {time.monotonic() - env_build_start:.1f}s")

    if source_model:
        src = MODELS_DIR / f"{source_model}.zip"
        if not src.exists():
            raise FileNotFoundError(f"Source model not found: {src}")
        hb(f"[cmd_train] {name}: TRANSFER from {src.name}")
        model = load_ppo_for_transfer(src, env, tensorboard_log=tb_log)
        reset_timesteps = False
    else:
        hb(f"[cmd_train] {name}: from scratch")
        model = make_ppo(env, tensorboard_log=tb_log)
        reset_timesteps = True

    hb(f"[cmd_train] starting model.learn() for {timesteps:,} timesteps "
       f"(N_ENVS={N_ENVS}, frame_stack={FRAME_STACK})")
    hb(f"[cmd_train] you should see heartbeat lines every ~30s during training")

    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=str(CHECKPOINTS_DIR),
        name_prefix=f"cpp_{size}x{size}_step",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # EvalCallback: periodically evaluates the model on a SEPARATE eval env
    # and keeps the BEST snapshot seen during training. PPO can regress after
    # convergence (large gradient variance once reward saturates), so taking
    # the final model is a known footgun. We use the best one instead.
    #
    # Settings reverted to Run 1 values (20 eps, 10 evals per stage). Earlier
    # attempts to tighten this (50 eps, 15 evals) caused regression — possibly
    # by introducing more noise in the "best snapshot" selection.
    best_dir = CHECKPOINTS_DIR / f"{name}_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    eval_env = make_single_env(size, obstacles, max_steps, render_mode=None,
                               stack=True)
    eval_freq_per_env = max(timesteps // (N_ENVS * 10), 25_000 // N_ENVS)
    eval_cb = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=20,                  # back to Run 1 setting
        eval_freq=eval_freq_per_env,
        best_model_save_path=str(best_dir),
        log_path=str(best_dir),
        deterministic=True,
        render=False,
        verbose=1,
    )

    # HeartbeatCallback: the SB3 progress bar (tqdm) updates in-place via \r
    # on a single line. If training freezes, the bar simply stops moving with
    # no separate "still alive" line. This callback prints one fresh line every
    # 30s with current step / ETA so you can detect a real freeze quickly.
    heartbeat_cb = HeartbeatCallback(every_seconds=30, total_timesteps=timesteps)

    learn_start = time.monotonic()
    model.learn(
        total_timesteps=timesteps,
        callback=[ckpt_cb, eval_cb, heartbeat_cb],
        reset_num_timesteps=reset_timesteps,
        progress_bar=True,
    )
    learn_elapsed = time.monotonic() - learn_start
    hb(f"[cmd_train] model.learn() finished in "
       f"{int(learn_elapsed) // 60}m{int(learn_elapsed) % 60:02d}s")
    env.close()

    # Pick the model that will go through final 100-episode evaluation.
    # We compare:
    #   (a) the snapshot saved by EvalCallback (highest mean coverage seen
    #       during training, evaluated on a 20-episode probe), AND
    #   (b) the final model state at end of model.learn().
    # We do a 50-episode shootout between them and pick the one with higher
    # mean coverage. This protects against:
    #   - EvalCallback locking in a "lucky" snapshot that doesn't generalize
    #   - The final model being a regressed version (PPO can drift after
    #     convergence due to gradient noise)
    final_model = model  # state at end of training
    best_model_zip = best_dir / "best_model.zip"
    if best_model_zip.exists():
        hb(f"[cmd_train] comparing best snapshot vs final model "
           f"(50 episodes each)...")
        best_snapshot = RecurrentPPO.load(str(best_model_zip), device="cpu")
        snap_summary = evaluate(
            best_snapshot, size, obstacles, max_steps,
            n_episodes=50, save_csv=None,
        )
        final_summary = evaluate(
            final_model, size, obstacles, max_steps,
            n_episodes=50, save_csv=None,
        )
        hb(f"[cmd_train]   best-snapshot: mean cov "
           f"{snap_summary['mean_coverage']:.2%} "
           f"(std {snap_summary['std_coverage']:.2%})")
        hb(f"[cmd_train]   final-model  : mean cov "
           f"{final_summary['mean_coverage']:.2%} "
           f"(std {final_summary['std_coverage']:.2%})")
        if snap_summary['mean_coverage'] >= final_summary['mean_coverage']:
            hb(f"[cmd_train] -> using BEST snapshot")
            model = best_snapshot
        else:
            hb(f"[cmd_train] -> using FINAL model (best snapshot was worse)")
            model = final_model
    else:
        hb(f"[cmd_train] no best snapshot saved; using final model")
        model = final_model
    eval_env.close()

    timestamped_path = MODELS_DIR / f"{name}.zip"
    model.save(str(timestamped_path))
    hb(f"[cmd_train] snapshot saved to {timestamped_path.name}")

    summary = evaluate(
        model, size, obstacles, max_steps,
        save_csv=RESULTS_DIR / f"{name}_eval.csv",
    )
    print_eval_summary(name, size, obstacles, summary,
                       model_path=str(timestamped_path))
    if register_as_official:
        approved, approved_path = _save_if_approved(model, size, summary)
    else:
        # Only check approval status; do NOT register as the official model.
        approved = summary["mean_coverage"] > APPROVAL_THRESHOLD
        approved_path = None
        if approved:
            print(f"  [APPROVED-but-unofficial] mean coverage "
                  f"{summary['mean_coverage']:.2%} > "
                  f"{APPROVAL_THRESHOLD:.0%} (not registered as "
                  f"cpp_{size}x{size}_approved.zip per request)")

    return StageResult(
        name=name, size=size, obstacles=obstacles, timesteps=timesteps,
        mean_coverage=summary["mean_coverage"],
        std_coverage=summary["std_coverage"],
        full_coverage_rate=summary["full_coverage_rate"],
        mean_reward=summary["mean_reward"],
        std_reward=summary["std_reward"],
        mean_steps=summary["mean_steps"],
        best_episode_idx=summary["best_episode_idx"],
        best_episode_coverage=summary["best_episode_coverage"],
        best_episode_reward=summary["best_episode_reward"],
        best_episode_steps=summary["best_episode_steps"],
        approved=approved,
        model_path=approved_path or str(timestamped_path),
    )


def cmd_curriculum(start_size: int, start_obstacles: int,
                   start_max_steps: int, start_timesteps: int) -> None:
    """Full curriculum: stage 1 from scratch on (start_size, start_obstacles, ...).
    Stages 2 and 3 transfer from the previous stage with scaled parameters."""
    print("\n" + "=" * 64)
    print(" CURRICULUM:  5x5 -> 10x10 -> 20x20")
    print("=" * 64)

    stages = [(start_size, start_obstacles, start_max_steps, start_timesteps)]
    for s_mult, o_mult, m_mult, t_mult in CURRICULUM_SCALING:
        stages.append((
            int(start_size * s_mult),
            int(start_obstacles * o_mult),
            int(start_max_steps * m_mult),
            int(start_timesteps * t_mult),
        ))

    history: list[StageResult] = []
    prev_approved: str | None = None

    for stage_idx, (size, obstacles, max_steps, timesteps) in enumerate(stages, 1):
        approved_name = f"cpp_{size}x{size}_approved"
        approved_path = MODELS_DIR / f"{approved_name}.zip"

        print(f"\n--- Stage {stage_idx}/{len(stages)}: {size}x{size}, "
              f"{obstacles} obstacles, max_steps={max_steps}, "
              f"timesteps={timesteps:,} ---")

        if approved_path.exists():
            print(f"    [skip] {approved_name}.zip already exists "
                  f"(delete it to re-train this stage)")
            prev_approved = approved_name
            continue

        res = cmd_train(
            size=size, obstacles=obstacles,
            max_steps=max_steps, timesteps=timesteps,
            source_model=prev_approved,
            save_name=approved_name,
        )
        history.append(res)

        if not res.approved:
            print(f"\n[curriculum] Stage {stage_idx} ({size}x{size}) FAILED.")
            print(f"             Mean coverage was {res.mean_coverage:.2%} "
                  f"(< {APPROVAL_THRESHOLD:.0%}).")
            print(f"             Previous approved models are kept intact.")
            print(f"             Adjust hyperparameters and re-run curriculum to retry.")
            break

        prev_approved = approved_name

    # Summary
    print("\n" + "=" * 64)
    print(" CURRICULUM SUMMARY")
    print("=" * 64)
    if not history:
        print("  (all stages already had approved models — nothing to do)")
    for r in history:
        flag = "APPROVED" if r.approved else "FAILED  "
        print(f"  [{flag}] {r.size:>2}x{r.size:<2}: mean cov "
              f"{r.mean_coverage:.2%} +/- {r.std_coverage:.2%},  "
              f"full {r.full_coverage_rate:5.1%},  "
              f"mean reward {r.mean_reward:6.2f},  "
              f"mean steps {r.mean_steps:5.1f},  "
              f"best ep coverage {r.best_episode_coverage:.2%}")
    print("=" * 64)

    if history:
        agg_path = RESULTS_DIR / f"curriculum_summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(agg_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(history[0]).keys()))
            w.writeheader()
            for r in history:
                w.writerow(asdict(r))
        print(f"  Aggregate summary saved to {agg_path}")


def cmd_bigtwenty() -> None:
    """Internal obstacle curriculum for the 20x20 grid only.

    Strategy: instead of training the 20x20 directly with 48 obstacles
    (which saturated at ~84% in earlier runs), we split the training into
    three sub-stages with progressively more obstacles, transferring
    between them. The starting point is the v1's 15x15 approved model.

    Sub-stages:
      1) 20x20 with 16 obstacles (4% block rate, similar density to 15x15)
         transfer from cpp_15x15_approved
      2) 20x20 with 32 obstacles (8% block rate)
         transfer from sub-stage 1
      3) 20x20 with 48 obstacles (12% block rate, the assignment target)
         transfer from sub-stage 2

    Final evaluation is done with 48 obstacles (the assignment level).
    Approval saved as cpp_20x20_approved.zip (with backup of any existing).
    """
    print("\n" + "=" * 64)
    print(" BIGTWENTY: internal obstacle curriculum for the 20x20 grid")
    print("=" * 64)

    SOURCE_MODEL = "cpp_15x15_approved"  # base for first sub-stage
    GRID_SIZE = 20
    MAX_STEPS = 1500

    # Confirm source model exists
    source_path = MODELS_DIR / f"{SOURCE_MODEL}.zip"
    if not source_path.exists():
        print(f"\n[bigtwenty] ERROR: source model {source_path} not found.")
        print(f"             You need cpp_15x15_approved.zip in models/ "
              f"to run bigtwenty.")
        print(f"             Restore from your backup or train the 15x15 first.")
        return

    sub_stages = [
        # (label, obstacles, timesteps)
        ("easy",    16, 2_500_000),
        ("medium",  32, 2_500_000),
        ("hard",    48, 3_000_000),
    ]

    history: list[StageResult] = []
    prev_save_name = SOURCE_MODEL

    for idx, (label, obs, timesteps) in enumerate(sub_stages, 1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"cpp_20x20_bigtwenty_{label}_{timestamp}"

        print(f"\n--- BigTwenty sub-stage {idx}/3: {label} "
              f"({obs} obstacles, {timesteps:,} timesteps) ---")
        print(f"    transfer from: {prev_save_name}")

        # Only the FINAL sub-stage (the one with the assignment's 48 obstacles)
        # competes for the official cpp_20x20_approved slot. Intermediate
        # sub-stages (16 and 32 obstacles) are easier configurations and
        # would unfairly displace the official model if they happened to pass
        # the threshold — they don't represent the assignment's actual task.
        is_final = (idx == len(sub_stages))

        res = cmd_train(
            size=GRID_SIZE,
            obstacles=obs,
            max_steps=MAX_STEPS,
            timesteps=timesteps,
            source_model=prev_save_name,
            save_name=save_name,
            register_as_official=is_final,
        )
        history.append(res)
        prev_save_name = save_name

        # If we are at the final sub-stage AND it's approved, register it
        # as the official cpp_20x20_approved model.
        if idx == len(sub_stages):
            print(f"\n[bigtwenty] final sub-stage finished.")
            print(f"  Mean coverage on 48 obstacles: {res.mean_coverage:.2%}")
            if res.approved:
                # cmd_train already saved as cpp_20x20_approved when approved=True
                # (since the approval check uses size 20, it triggers the alias).
                print(f"  [APPROVED] saved to cpp_20x20_approved.zip")
            else:
                print(f"  [NOT APPROVED] mean coverage {res.mean_coverage:.2%} "
                      f"<= {APPROVAL_THRESHOLD:.0%}")
                print(f"  Existing cpp_20x20_approved.zip (if any) was kept intact.")

    # Summary
    print("\n" + "=" * 64)
    print(" BIGTWENTY SUMMARY")
    print("=" * 64)
    for label_idx, (label, obs, _) in enumerate(sub_stages):
        if label_idx >= len(history):
            break
        r = history[label_idx]
        flag = "APPROVED" if r.approved else "FAILED  "
        print(f"  [{flag}] {label:>6} ({obs:>2} obs): mean cov "
              f"{r.mean_coverage:.2%} +/- {r.std_coverage:.2%},  "
              f"full {r.full_coverage_rate:5.1%},  "
              f"mean reward {r.mean_reward:6.2f},  "
              f"mean steps {r.mean_steps:6.1f},  "
              f"best ep coverage {r.best_episode_coverage:.2%}")
    print("=" * 64)

    if history:
        agg_path = RESULTS_DIR / f"bigtwenty_summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(agg_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(history[0]).keys()))
            w.writeheader()
            for r in history:
                w.writerow(asdict(r))
        print(f"  Aggregate summary saved to {agg_path}")





def _resolve_model_path(size: int) -> Path:
    src = MODELS_DIR / f"cpp_{size}x{size}_approved.zip"
    if not src.exists():
        raise FileNotFoundError(
            f"No approved model for {size}x{size} found at {src}. "
            f"Train it first with `train {size} <obstacles> <max_steps> <timesteps>`."
        )
    return src


def cmd_test(size: int, obstacles: int) -> None:
    src = _resolve_model_path(size)
    max_steps = {5: 200, 10: 400, 20: 1200}.get(size, max(200, size * size * 3))
    env = make_single_env(size, obstacles, max_steps, render_mode=None, stack=True)
    model = RecurrentPPO.load(str(src), env=env, device="cpu")
    summary = evaluate(
        model, size, obstacles, max_steps,
        save_csv=RESULTS_DIR / f"{src.stem}_test_{datetime.now():%Y%m%d_%H%M%S}.csv",
    )
    env.close()
    print_eval_summary(src.stem, size, obstacles, summary, model_path=str(src))


def cmd_run(size: int, obstacles: int) -> None:
    src = _resolve_model_path(size)
    max_steps = {5: 200, 10: 400, 20: 1200}.get(size, max(200, size * size * 3))
    env = make_single_env(size, obstacles, max_steps, render_mode="human", stack=True)
    model = RecurrentPPO.load(str(src), env=env, device="cpu")
    obs = env.reset()
    done = False
    total_reward = 0.0
    last_info: dict = {}
    step = 0
    # LSTM hidden state propagation
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts,
            deterministic=True,
        )
        obs, reward, dones, infos = env.step(action)
        total_reward += float(reward[0])
        last_info = infos[0]
        done = bool(dones[0])
        episode_starts = np.zeros((1,), dtype=bool)
        step += 1
    env.close()
    print(f"\nDone in {step} steps  |  "
          f"coverage = {last_info.get('coverage_ratio', 0.0):.1%}  |  "
          f"reward = {total_reward:.2f}")


# --------------------------------------------------------------------------- cli

def main() -> None:
    parser = argparse.ArgumentParser(prog="train_grid_world_cpp.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train (or transfer) on a single grid")
    p_train.add_argument("size", type=int)
    p_train.add_argument("obstacles", type=int)
    p_train.add_argument("max_steps", type=int)
    p_train.add_argument("timesteps", type=int)
    p_train.add_argument("--from", dest="from_model", type=str, default=None,
                         help="basename in models/ to start from (transfer learning)")
    p_train.add_argument("--save-name", type=str, default=None)

    p_curr = sub.add_parser("curriculum", help="run full pipeline 5x5->10x10->20x20")
    p_curr.add_argument("size", type=int)
    p_curr.add_argument("obstacles", type=int)
    p_curr.add_argument("max_steps", type=int)
    p_curr.add_argument("timesteps", type=int)

    sub.add_parser("bigtwenty", help="internal obstacle curriculum for the "
                   "20x20 grid (16 -> 32 -> 48 obstacles), starting from "
                   "the existing cpp_15x15_approved model. ~4h on 7700x.")




    p_test = sub.add_parser("test", help="evaluate the approved model for a size")
    p_test.add_argument("size", type=int)
    p_test.add_argument("obstacles", type=int)

    p_run = sub.add_parser("run", help="visualize 1 episode of the approved model")
    p_run.add_argument("size", type=int)
    p_run.add_argument("obstacles", type=int)

    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args.size, args.obstacles, args.max_steps, args.timesteps,
                  source_model=args.from_model, save_name=args.save_name)
    elif args.cmd == "curriculum":
        cmd_curriculum(args.size, args.obstacles, args.max_steps, args.timesteps)
    elif args.cmd == "bigtwenty":
        cmd_bigtwenty()
    elif args.cmd == "test":
        cmd_test(args.size, args.obstacles)
    elif args.cmd == "run":
        cmd_run(args.size, args.obstacles)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
