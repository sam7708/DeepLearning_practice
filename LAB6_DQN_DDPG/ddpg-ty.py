"""DLP DDPG Lab"""
__author__ = 'chengscott'
__copyright__ = 'Copyright 2019, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class OrnsteinUhlenbeckProcess:
  """1-dimension Ornstein-Uhlenbeck process"""
  def sample(self, mu=0, std=.2, theta=.15, dt=1e-2, sqrt_dt=1e-1):
    self.x += theta * (mu - self.x) * dt + std * sqrt_dt * random.gauss(0, 1)
    return self.x

  def reset(self, x0=0):
    self.x = x0


class ReplayMemory:
  def __init__(self, capacity):
    self._buffer = deque(maxlen=capacity)

  def __len__(self):
    return len(self._buffer)

  def append(self, *transition):
    # (state, action, reward, next_state, done)
    self._buffer.append(tuple(map(tuple, transition)))

  def sample(self, batch_size=1):
    return random.sample(self._buffer, batch_size)


class ActorNet(nn.Module):
  def __init__(self, state_dim=3, action_dim=1, hidden_dim=(400, 300)):
    super().__init__()
    ## TODO ##
    h1, h2 = hidden_dim
    self.L1 = nn.Sequential(
        nn.Linear(state_dim, h1),
        nn.ReLU(),    
      )
    self.L2 = nn.Sequential(
        nn.Linear(h1, h2),
        nn.ReLU(),
      )
    self.L3 = nn.Sequential(
        nn.Linear(h2, action_dim)
      )
    #raise NotImplementedError

  def forward(self, x):
    ## TODO ##
    x = self.L1(x)
    x = self.L2(x)
    x = self.L3(x)
    return x
    #raise NotImplementedError


class CriticNet(nn.Module):
  def __init__(self, state_dim=3, action_dim=1, hidden_dim=(400, 300)):
    super().__init__()
    h1, h2 = hidden_dim
    self.critic_head = nn.Sequential(
        nn.Linear(state_dim, h1),
        nn.ReLU(),
    )
    self.critic = nn.Sequential(
        nn.Linear(h1 + action_dim, h2),
        nn.ReLU(),
        nn.Linear(h2, action_dim),
    )

  def forward(self, x, action):
    x = self.critic_head(x)
    return self.critic(torch.cat([x, action], dim=1))


def select_action(state, low=-2, high=2):
  """based on the behavior (actor) network and exploration noise"""
  random_process = OrnsteinUhlenbeckProcess()
  random_process.reset()
  action = actor_net(state)
  noise = random_process.sample()
  actionnoise = action + noise
  actionnoise = actionnoise.item()
  return max(min(actionnoise, high), low)
  #raise NotImplementedError


def update_behavior_network():
  def transitions_to_tensors(transitions, device=args.device):
    """convert a batch of transitions to tensors"""
    return (torch.Tensor(x).to(device) for x in zip(*transitions))

  # sample a minibatch of transitions
  transitions = memory.sample(args.batch_size)
  state, action, reward, state_next, done = transitions_to_tensors(transitions)

  ## update critic ##
  q_value = critic_net.forward(state, action)
  with torch.no_grad():
    a_next = actor_net.forward(state_next)
    q_next = target_critic_net.forward(state_next,a_next)
  q_next = (q_next*args.gamma+reward) * abs(1-done)
  critic_loss = F.smooth_l1_loss(q_value, q_next)
  actor_net.zero_grad()
  critic_net.zero_grad()
  critic_loss.backward()
  critic_opt.step()

  actor_loss = -critic_net.forward(state, actor_net.forward(state)).mean()
  actor_net.zero_grad()
  critic_net.zero_grad()
  actor_loss.backward()
  actor_opt.step()

def plot_result(episode_reward_list):
  plt.plot(episode_reward_list)
  plt.ylabel('Reward')
  plt.xlabel('Episode')
  plt.title('Training Process')
  plt.show()

def update_target_network(target_net, net):
  tau = args.tau
  for target, behavior in zip(target_net.parameters(), net.parameters()):
    ## TODO ##
    target.data.copy_(behavior.data * tau + target.data * (1.0 - tau))
    #raise NotImplementedError


def train(env):
  print('Start Training')
  total_steps = 0
  episode_reward_list = []
  for episode in range(args.episode):
    total_reward = 0
    random_process.reset()
    state = env.reset()
    for t in itertools.count(start=1):
      # select action
      if total_steps < args.warmup:
        action = float(env.action_space.sample())
      else:
        state_tensor = torch.Tensor(state).to(args.device)
        action = select_action(state_tensor)
      # execute action
     # print(action.item())
      next_state, reward, done, _ = env.step([action])
      # store transition
      memory.append(state, [action], [reward], next_state, [int(done)])
      if total_steps >= args.warmup:
        # update the behavior networks
        update_behavior_network()
        # update the target networks
        update_target_network(target_actor_net, actor_net)
        update_target_network(target_critic_net, critic_net)

      state = next_state
      total_reward += reward
      total_steps += 1
      if done:
        print('Step: {}\tEpisode: {}\tLength: {}\tTotal reward: {}'.format(
            total_steps, episode, t, total_reward))
        episode_reward_list.append(total_reward)
        break
  env.close()
  plot_result(episode_reward_list)


def test(env, render):
  print('Start Testing')
  seeds = (20190813 + i for i in range(10))
  result = 0.0
  total_reward = 0
  for i in range(10):
    for seed in seeds:
      total_reward = 0
      env.seed(seed)
      state = env.reset()
      done = False
      while not done:
        env.render()
        state_tensor = torch.Tensor(state).to(args.device)
        action = select_action(state_tensor)
        #print(action)
        state, reward, done, info = env.step([action])
        total_reward +=reward
    print(total_reward)    
    result += total_reward*0.1
    ## TODO ##
  print(result)
    #raise NotImplementedError
  env.close()


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-d', '--device', default='cuda')
  # network
  parser.add_argument('-m', '--model', default='pendulum_model')
  parser.add_argument('--restore', action='store_true')
  # train
  parser.add_argument('-e', '--episode', default=550, type=int)
  parser.add_argument('-c', '--capacity', default=10000, type=int)
  parser.add_argument('-bs', '--batch_size', default=64, type=int)
  parser.add_argument('--warmup', default=10000, type=int)
  parser.add_argument('--lra', default=1e-4, type=float)
  parser.add_argument('--lrc', default=1e-3, type=float)
  parser.add_argument('--gamma', default=.99, type=float)
  parser.add_argument('--tau', default=.001, type=float)
  # test
  parser.add_argument('--render', action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  # environment
  env = gym.make('Pendulum-v0')
  # behavior network
  actor_net = ActorNet().to(args.device)
  critic_net = CriticNet().to(args.device)
  if not args.restore:
    # target network
    target_actor_net = ActorNet().to(args.device)
    target_critic_net = CriticNet().to(args.device)
    # initialize target network
    target_actor_net.load_state_dict(actor_net.state_dict())
    target_critic_net.load_state_dict(critic_net.state_dict())
    actor_opt = torch.optim.Adam(actor_net.parameters(), lr=args.lra)
    critic_opt = torch.optim.Adam(critic_net.parameters(), lr=args.lrc)
    #raise NotImplementedError
    # random process
    random_process = OrnsteinUhlenbeckProcess()
    # memory
    memory = ReplayMemory(capacity=args.capacity)
    # train
    train(env)
    # save model
    torch.save(
        {
            'actor': actor_net.state_dict(),
            'critic': critic_net.state_dict(),
        }, args.model)
  # load model
  model = torch.load(args.model)
  actor_net.load_state_dict(model['actor'])
  critic_net.load_state_dict(model['critic'])
  # test
  test(env, args.render)
