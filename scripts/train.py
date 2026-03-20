#!/usr/bin/env python
"""Train a tabular Markov Q-learning agent for 2048."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import FeatureQAgent, MarkovQAgent
from src.env import Gym2048Env
from src.utils import MLFlowLogger, ModelCheckpoint


def evaluate_agent(
    agent: MarkovQAgent,
    env: Gym2048Env,
    n_episodes: int = 10,
    max_steps: int = 10000,
) -> dict:
    """Evaluate the agent without exploration."""
    scores = []
    max_tiles = []
    steps_list = []

    for _ in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_steps = 0
        valid_actions = info["valid_actions"]

        while not done and episode_steps < max_steps:
            action = agent.select_action(state, valid_actions=valid_actions, use_epsilon=False)
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            valid_actions = info["valid_actions"]
            episode_steps += 1

        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        steps_list.append(episode_steps)

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "mean_max_tile": float(np.mean(max_tiles)),
        "std_max_tile": float(np.std(max_tiles)),
        "mean_steps": float(np.mean(steps_list)),
        "best_score": int(np.max(scores)),
        "best_max_tile": int(np.max(max_tiles)),
    }


def train(
    n_episodes: int = 5000,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 4000,  # Can be int (linear episodes) or float (exponential rate)
    eval_freq: int = 250,
    checkpoint_freq: int = 1000,
    reward_mode: str = "score",
    invalid_move_penalty: float = 0.0,
    max_steps: int = 10000,
    save_dir: str = "./models",
    experiment_name: str = "2048-markov-q",
    agent_type: str = "markov",
    seed: int | None = None,
):
    """Main training loop."""
    print("=" * 70)
    print(f"Training {agent_type.upper()} Q-Learning Agent for 2048")
    print("=" * 70)

    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed: {seed}")

    env = Gym2048Env(
        reward_mode=reward_mode,
        invalid_move_penalty=invalid_move_penalty,
        max_steps=max_steps,
    )

    # Create agent based on type
    if agent_type == "feature":
        agent = FeatureQAgent(
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )
    else:  # markov (full state)
        agent = MarkovQAgent(
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )

    print(f"Environment: {env.__class__.__name__}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    print(f"Reward mode: {reward_mode}")
    print(f"Agent: {agent.__class__.__name__}")

    # Set checkpoint prefix based on agent type
    prefix = "feature_q" if agent_type == "feature" else "markov_q"

    checkpoint_manager = ModelCheckpoint(
        save_dir=save_dir,
        filename_prefix=prefix,
        save_best=True,
        mode="max",
        verbose=True,
    )

    config = {
        "agent_type": agent_type,
        "n_episodes": n_episodes,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "reward_mode": reward_mode,
        "invalid_move_penalty": invalid_move_penalty,
        "max_steps": max_steps,
        "seed": seed,
    }
    checkpoint_manager.save_config(config)

    logger = MLFlowLogger(experiment_name=experiment_name, run_name=f"markov_q_{n_episodes}ep")

    with logger:
        logger.log_params(config)

        best_eval_score = float("-inf")
        episode_scores = []
        pbar = tqdm(range(n_episodes), desc="Training")

        for episode in pbar:
            episode_seed = seed + episode if seed is not None else None
            state, info = env.reset(seed=episode_seed)
            valid_actions = info["valid_actions"]
            done = False
            episode_reward = 0.0
            episode_td_error = 0.0
            episode_q_values = 0.0
            step_count = 0

            while not done:
                action = agent.select_action(state, valid_actions=valid_actions, use_epsilon=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                td_error, q_value = agent.learn(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    next_valid_actions=info["valid_actions"],
                )

                episode_reward += reward
                episode_td_error += td_error
                episode_q_values += q_value
                step_count += 1

                state = next_state
                valid_actions = info["valid_actions"]

            agent.update_epsilon()
            episode_scores.append(info["score"])

            avg_td_error = episode_td_error / step_count if step_count > 0 else 0.0
            avg_q_value = episode_q_values / step_count if step_count > 0 else 0.0

            logger.log_metrics(
                {
                    "episode_reward": episode_reward,
                    "episode_score": info["score"],
                    "episode_max_tile": info["max_tile"],
                    "episode_steps": step_count,
                    "epsilon": agent.epsilon,
                    "td_error": avg_td_error,
                    "q_value": avg_q_value,
                    "num_states": len(agent),
                },
                step=episode,
            )

            pbar.set_postfix(
                {
                    "score": info["score"],
                    "max_tile": info["max_tile"],
                    "epsilon": f"{agent.epsilon:.3f}",
                    "states": len(agent),
                }
            )

            if (episode + 1) % eval_freq == 0:
                print(f"\n--- Evaluation at Episode {episode + 1} ---")
                eval_metrics = evaluate_agent(agent, env, n_episodes=10, max_steps=max_steps)
                mean_score = eval_metrics["mean_score"]
                std_score = eval_metrics["std_score"]
                print(f"Mean Score: {mean_score:.2f} ± {std_score:.2f}")
                print(f"Mean Max Tile: {eval_metrics['mean_max_tile']:.2f}")
                print(f"Best Score: {eval_metrics['best_score']}")
                print(f"Best Max Tile: {eval_metrics['best_max_tile']}")

                logger.log_metrics(
                    {
                        "eval_mean_score": eval_metrics["mean_score"],
                        "eval_std_score": eval_metrics["std_score"],
                        "eval_mean_max_tile": eval_metrics["mean_max_tile"],
                        "eval_best_score": eval_metrics["best_score"],
                        "eval_best_max_tile": eval_metrics["best_max_tile"],
                    },
                    step=episode,
                )

                if eval_metrics["mean_score"] > best_eval_score:
                    best_eval_score = eval_metrics["mean_score"]
                    checkpoint_manager.save(
                        agent,
                        epoch=episode + 1,
                        metric=best_eval_score,
                        metadata={"num_states": len(agent), "eval_metrics": eval_metrics},
                        is_best=True,
                    )

            if (episode + 1) % checkpoint_freq == 0:
                checkpoint_manager.save(
                    agent,
                    epoch=episode + 1,
                    metric=float(np.mean(episode_scores[-eval_freq:])),
                    metadata={"num_states": len(agent), "epsilon": agent.epsilon},
                )

        print("\n" + "=" * 70)
        print("Final Evaluation (50 episodes)")
        print("=" * 70)
        final_eval = evaluate_agent(agent, env, n_episodes=50, max_steps=max_steps)
        print(f"Mean Score: {final_eval['mean_score']:.2f} ± {final_eval['std_score']:.2f}")
        print(f"Mean Max Tile: {final_eval['mean_max_tile']:.2f}")
        print(f"Best Score: {final_eval['best_score']}")
        print(f"Best Max Tile: {final_eval['best_max_tile']}")

        logger.log_metrics(
            {
                "final_mean_score": final_eval["mean_score"],
                "final_std_score": final_eval["std_score"],
                "final_mean_max_tile": final_eval["mean_max_tile"],
                "final_best_score": final_eval["best_score"],
                "final_best_max_tile": final_eval["best_max_tile"],
            }
        )

        checkpoint_manager.save(
            agent,
            epoch=n_episodes,
            metric=final_eval["mean_score"],
            metadata={"num_states": len(agent), "final_eval": final_eval},
        )

        print("\nTraining complete!")
        print(f"Models saved to: {save_dir}")
        print("MLflow logs: ./experiments/mlruns")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train a Q-learning agent for 2048")
    parser.add_argument(
        "--agent-type",
        type=str,
        default="markov",
        choices=["markov", "feature"],
        help="Agent type: 'markov' (full state) or 'feature' (minimal features)",
    )
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to train")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=4000,
        help="Epsilon decay (episodes for linear, rate for exponential)",
    )
    parser.add_argument("--eval-freq", type=int, default=250, help="Episodes between evaluations")
    parser.add_argument(
        "--checkpoint-freq", type=int, default=1000, help="Episodes between checkpoints"
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="score",
        choices=["score", "log_score", "max_tile"],
        help="Reward shaping mode",
    )
    parser.add_argument(
        "--invalid-move-penalty",
        type=float,
        default=0.0,
        help="Penalty applied to invalid moves",
    )
    parser.add_argument("--max-steps", type=int, default=10000, help="Episode step limit")
    parser.add_argument("--save-dir", type=str, default="./models", help="Checkpoint directory")
    parser.add_argument(
        "--experiment",
        type=str,
        default="2048-markov-q",
        help="MLflow experiment name",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()
    train(
        n_episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        reward_mode=args.reward_mode,
        invalid_move_penalty=args.invalid_move_penalty,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        experiment_name=args.experiment,
        agent_type=args.agent_type,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
