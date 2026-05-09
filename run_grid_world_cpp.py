"""
run_grid_world_cpp.py
---------------------
Runs ONE episode of the CPP environment with a RANDOM agent (sanity check
of the environment and observation pipeline).

Usage:
    python run_grid_world_cpp.py                      # 5x5, 3 obstacles, with rendering
    python run_grid_world_cpp.py --headless           # 5x5, no rendering (faster)
    python run_grid_world_cpp.py 10 12 400            # 10x10 with 12 obstacles, 400 max steps
    python run_grid_world_cpp.py 20 48 1200           # 20x20 with 48 obstacles
"""

import argparse

import gymnasium as gym

import gymnasium_env  # noqa: F401  registers GridWorldCPP-v0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int, nargs="?", default=5)
    parser.add_argument("obstacles", type=int, nargs="?", default=3)
    parser.add_argument("max_steps", type=int, nargs="?", default=200)
    parser.add_argument("--headless", action="store_true",
                        help="disable pygame rendering (faster)")
    args = parser.parse_args()

    render_mode = None if args.headless else "human"

    env = gym.make(
        "GridWorldCPP-v0",
        size=args.size,
        obs_quantity=args.obstacles,
        max_steps=args.max_steps,
        render_mode=render_mode,
    )

    obs, info = env.reset(seed=42)
    print("=" * 56)
    print(" GridWorld CPP - Random Agent")
    print(f" Grid       : {args.size}x{args.size}")
    print(f" Obstacles  : {args.obstacles}")
    print(f" Accessible : {info['accessible_cells']} cells")
    print("=" * 56)

    total_reward = 0.0
    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step += 1
        done = terminated or truncated
        tag = "NEW   " if reward > 0.4 else ("DONE  " if terminated else "       ")
        print(f"  step {step:4d} | a={action} | r={reward:+.2f} "
              f"| cov={info['coverage_ratio']:6.1%} "
              f"({info['covered_cells']:3d}/{info['accessible_cells']:3d}) {tag}")

    print("=" * 56)
    print(f" Episode finished after {step} steps")
    print(f" Total reward    : {total_reward:.2f}")
    print(f" Final coverage  : {info['coverage_ratio']:.1%} "
          f"({info['covered_cells']}/{info['accessible_cells']} cells)")
    print(f" {'OK: full coverage' if info['coverage_ratio'] == 1.0 else 'NO full coverage (random agent expected)'}")
    print("=" * 56)
    env.close()


if __name__ == "__main__":
    main()
