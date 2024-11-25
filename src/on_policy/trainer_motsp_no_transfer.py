"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable
import yaml

from model import DRL4TSP, Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output


def prepare(obs, static_size, dynamic_size, sequence_size):
    static = obs[:, :static_size].reshape(-1, static_size, sequence_size)
    dynamic = obs[:, static_size:].reshape(-1, dynamic_size, sequence_size)
    return static, dynamic


def validate(
    data_loader, actor, reward_fn, w1, w2, render_fn=None, save_dir=".", num_plot=5
):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    obj1s = []
    obj2s = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward, obj1, obj2 = reward_fn(static, tour_indices, w1, w2)

        rewards.append(torch.mean(reward.detach()).item())
        obj1s.append(torch.mean(obj1.detach()).item())
        obj2s.append(torch.mean(obj2.detach()).item())
        if render_fn is not None and batch_idx < num_plot:
            name = "batch%d_%2.4f.png" % (batch_idx, torch.mean(reward.detach()).item())
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards), np.mean(obj1s), np.mean(obj2s)


def train(
    actor,
    critic,
    env,
    env_args,
    task,
    num_nodes,
    batch_size,
    actor_lr,
    critic_lr,
    max_grad_norm,
    **kwargs,
):
    """Constructs the main actor & critic networks, and performs all training."""

    now = "%s" % datetime.datetime.now().time()
    now = now.replace(":", "_")
    bname = "_4static"
    save_dir = os.path.join(task + bname, "%d" % num_nodes, now)

    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    best_params = None
    best_reward = np.inf

    train_slot = 0

    for epoch in range(5):
        print("epoch %d start:" % epoch)
        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []
        objs = []

        epoch_start = time.time()
        start = epoch_start

        env_info = env.get_env_info(True)
        env.reset()
        for slot in range(env_info["episode_limit"]):
            train_slot += 1

            obs = env.get_luav_obs_new()
            static, dynamic, x0 = prepare(
                obs, int(env_info["l_s_obs_shape"] / env_args["l_s_obs_shape"])
            )

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)

            (
                reward,
                f_reward_agents,
                f_reward,
                slot_end,
                terminated,
                env_info,
            ) = env.step(0, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = reward - critic_est
            advantage_show = advantage.detach()
            tour_logp_show = tour_logp.sum(dim=1)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage**2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())
            if (train_slot) % 200 == 0:
                # print("\n")
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])
                print(
                    "  reward: %2.3f, loss: %2.4f, took: %2.4fs"
                    % (
                        mean_reward,
                        mean_loss,
                        times[-1],
                    )
                )

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, "%s" % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, "actor.pt")
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, "critic.pt")
        torch.save(critic.state_dict(), save_path)

        # # Save rendering of validation set tours
        # valid_dir = os.path.join(save_dir, "%s" % epoch)

        # print("begin valid")
        # s = time.time()
        # mean_valid, mean_obj1_valid, mean_obj2_valid = validate(
        #     valid_loader, actor, reward_fn, w1, w2, render_fn, valid_dir, num_plot=5
        # )
        # print("valid end time: %2.4f" % (time.time() - s))
        # # Save best model parameters
        # if mean_valid < best_reward:

        #     best_reward = mean_valid

        #     # save_path = os.path.join(save_dir, 'actor.pt')
        #     # torch.save(actor.state_dict(), save_path)
        #     #
        #     # save_path = os.path.join(save_dir, 'critic.pt')
        #     # torch.save(critic.state_dict(), save_path)
        #     # 存在w_1_0主文件夹下，多存一份，用来transfer to next w
        #     main_dir = os.path.join(
        #         task + bname, "%d" % num_nodes, "w_%2.2f_%2.2f" % (w1, w2)
        #     )
        #     save_path = os.path.join(main_dir, "actor.pt")
        #     torch.save(actor.state_dict(), save_path)
        #     save_path = os.path.join(main_dir, "critic.pt")
        #     torch.save(critic.state_dict(), save_path)

        # print(
        #     "Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, obj1_valid: %2.3f, obj2_valid: %2.3f. took: %2.4fs "
        #     "(%2.4fs / 100 batches)\n"
        #     % (
        #         mean_loss,
        #         mean_reward,
        #         mean_valid,
        #         mean_obj1_valid,
        #         mean_obj2_valid,
        #         time.time() - epoch_start,
        #         np.mean(times),
        #     )
        # )


def train_vrp(
    config_file: str, args, w1=1, w2=0, checkpoint=None, reconfig: Callable = None
):

    from ..envs.env_4 import SNEnv

    with open(f"src/config/envs/{config_file}.yaml", "r", encoding="utf-8") as f:
        env_args = yaml.load(f, Loader=yaml.FullLoader)["env_args"]

    if reconfig is not None:
        reconfig(env_args)

    env = SNEnv(**env_args)
    env_info = env.get_env_info(True)

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = int(env_info["l_s_obs_shape"] / env_args["fuav_num"])  # (x, y)
    DYNAMIC_SIZE = int(
        env_info["l_d_obs_shape"] / env_args["fuav_num"]
    )  # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]
    print(max_load)

    actor = DRL4TSP(
        STATIC_SIZE,
        DYNAMIC_SIZE,
        args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    if not args.test:
        train(actor, critic, env, env_args, **kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Combinatorial Optimization")
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--checkpoint", default="tsp/20/w_1_0/20_06_30.888074")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--task", default="vrp")
    parser.add_argument("--nodes", dest="num_nodes", default=20, type=int)
    parser.add_argument("--actor_lr", default=5e-4, type=float)
    parser.add_argument("--critic_lr", default=5e-4, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--hidden", dest="hidden_size", default=128, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--layers", dest="num_layers", default=1, type=int)
    parser.add_argument("--train-size", default=500000, type=int)
    parser.add_argument("--valid-size", default=1000, type=int)

    args = parser.parse_args()

    if args.task == "vrp":
        train_vrp(args)
    else:
        raise ValueError("Task <%s> not understood" % args.task)
