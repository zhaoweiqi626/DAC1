import numpy as np
import torch
import gym
import argparse
import os
import copy
from pathlib import Path


from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic, RescaleAction
from tqc.functions import eval_policy


EPISODE_LENGTH = 1000


def main(args, results_dir, models_dir, prefix):
    # --- Init ---

    # remove TimeLimit
    env = gym.make(args.env).unwrapped
    eval_env = gym.make(args.env).unwrapped

    env = RescaleAction(env, -1., 1.)
    eval_env = RescaleAction(eval_env, -1., 1.)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = structures.ReplayBuffer(state_dim, action_dim)
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim, args.n_quantiles, args.n_nets).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets

    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=top_quantiles_to_drop,
                      discount=args.discount,
                      tau=args.tau,
                      target_entropy=-np.prod(env.action_space.shape).item())

    evaluations = []
    state, done = env.reset(), False
    episode_return = 0
    episode_timesteps = 0
    episode_num = 0

    actor.train()
    for t in range(int(args.max_timesteps)):
        action = actor.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_timesteps += 1

        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_return += reward

        # Train agent after collecting sufficient data
        if t >= args.batch_size:
            trainer.train(replay_buffer, args.batch_size)

        if done or episode_timesteps >= EPISODE_LENGTH:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_return:.3f}")
            # Reset environment
            state, done = env.reset(), False

            episode_return = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            file_name = f"{prefix}_{args.env}_{args.seed}"
            evaluations.append(eval_policy(actor, eval_env, EPISODE_LENGTH))
            np.save(results_dir / file_name, evaluations)
            if args.save_model: trainer.save(models_dir / file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Humanoid-v3")          # OpenAI gym environment name
    parser.add_argument("--eval_freq", default=1e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--log_dir", default='.')
    parser.add_argument("--prefix", default='')
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    results_dir = log_dir / 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    models_dir = log_dir / 'models'
    if args.save_model and not os.path.exists(models_dir):
        os.makedirs(models_dir)

    main(args, results_dir, models_dir, args.prefix)
