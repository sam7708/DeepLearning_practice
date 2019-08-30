"""DLP DQN Lab"""
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


class My_loss(nn.Module):
  def __init__(self):
    super().__init__()
        
  def forward(self, x, y):
    return torch.mean(torch.pow((x - y), 2))



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


class DQN(nn.Module):
  def __init__(self, state_dim=4, action_dim=2, hidden_dim=24):
    super().__init__()
    self.conv1 = nn.Linear(state_dim, hidden_dim)
    self.conv2 = nn.Linear(hidden_dim, hidden_dim)
    self.conv3 = nn.Linear(hidden_dim, action_dim)
    # Called with either one element to determine next action, or a batch

  def forward(self, x):
    x = torch.nn.functional.relu(self.conv1(x))
    x = torch.nn.functional.relu(self.conv2(x))
    x = self.conv3(x)
    return x 

def select_action(epsilon, state, action_dim=2):
  """epsilon-greedy based on behavior network"""
  sample = random.random()
  if sample > epsilon:
    return behavior_net(state).max(0)[1].view(1, 1).item()
  else:
    return random.randrange(env.action_space.n)


def update_behavior_network():
  def transitions_to_tensors(transitions, device=args.device):
    """convert a batch of transitions to tensors"""
    return (torch.Tensor(x).to(device) for x in zip(*transitions))

  # sample a minibatch of transitions
  transitions = memory.sample(args.batch_size)
  state, action, reward, next_state, done = transitions_to_tensors(transitions)
  q_value = behavior_net(state).gather(1, action.long())
  q_next = target_net(next_state).detach()*args.gamma+reward
  loss = F.smooth_l1_losscriterion(q_value, q_next*abs(1-done))
  # optimize
  optimizer.zero_grad()
  loss.backward()
  nn.utils.clip_grad_norm_(behavior_net.parameters(), 5)
  optimizer.step()

def plot_result(episode_reward_list):
  plt.plot(episode_reward_list)
  plt.ylabel('Reward')
  plt.xlabel('Episode')
  plt.title('Training Process')
  plt.show()

def train(env):
  print('Start Training')
  total_steps, epsilon = 0, 1.
  episode_reward_list = []
  for episode in range(args.episode):
    total_reward = 0
    state = env.reset()
    for t in itertools.count(start=1):
      # select action
      if total_steps < args.warmup:
        action = env.action_space.sample()
      else:
        state_tensor = torch.Tensor(state).to(args.device)
        action = select_action(epsilon, state_tensor)
        epsilon = max(epsilon * args.eps_decay, args.eps_min)
      # execute action
      if(t > 9990):
        print(action)
      next_state, reward, done, _ = env.step(action)
      # store transition
      memory.append(state, [action], [reward / 10], next_state, [int(done)])
      if total_steps >= args.warmup and total_steps % args.freq == 0:
        # update the behavior network
        update_behavior_network()
      if total_steps % args.target_freq == 0:
        # TODO: update the target network by copying from the behavior network
        target_net.load_state_dict(behavior_net.state_dict())
        #raise NotImplementedError

      state = next_state
      total_reward += reward
      total_steps += 1
      if done:
        print('Step: {}\tEpisode: {}\tTotal reward: {}\tEpsilon: {}'.format(
            total_steps, episode, total_reward, epsilon))
        episode_reward_list.append(total_reward)
        break

  env.close()
  plot_result(episode_reward_list)

def test(env, render):
  print('Start Testing')
  epsilon = args.test_epsilon
  seeds = (20190813 + i for i in range(10))
  episode_reward = 0.
  total_reward = 0.
  for i in range(10):
    for seed in seeds:
      total_reward = 0
      env.seed(seed)
      state = env.reset()
      ## TODO ##
      episode_steps = 0
      episode_reward = 0.
      done = False
      while not done:
        env.render()
        state_tensor = torch.Tensor(state).to(args.device)
        action = select_action(epsilon, state_tensor)
        state, reward, done, info = env.step(action)
        episode_reward += reward
        episode_steps += 1
      #raise NotImplementedError
    print(episode_reward)
    total_reward += episode_reward/10
  #print(total_reward)
  env.close()


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-d', '--device', default='cuda')
  # network
  parser.add_argument('-m', '--model', default='cartpole_model')
  parser.add_argument('--restore', action='store_true')
  # train
  parser.add_argument('-e', '--episode', default=1800, type=int)
  parser.add_argument('-c', '--capacity', default=10000, type=int)
  parser.add_argument('-bs', '--batch_size', default=128, type=int)
  parser.add_argument('--warmup', default=10000, type=int)
  parser.add_argument('--lr', default=.0005, type=float)
  parser.add_argument('--eps_decay', default=.995, type=float)
  parser.add_argument('--eps_min', default=.01, type=float)
  parser.add_argument('--gamma', default=.99, type=float)
  parser.add_argument('--freq', default=4, type=int)
  parser.add_argument('--target_freq', default=1000, type=int)
  # test
  parser.add_argument('--test_epsilon', default=.001, type=float)
  parser.add_argument('--render', action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  # environment
  env = gym.make('CartPole-v1')
  # behavior network
  behavior_net = DQN().to(args.device)
  if not args.restore:
    # target network
    target_net = DQN().to(args.device)
    # initialize target network
    target_net.load_state_dict(behavior_net.state_dict())
    # TODO: optimizer
    # optimizer = ?
    optimizer = torch.optim.Adam(behavior_net.parameters(), lr=args.lr)
    #criterion = nn.MSELoss()
    # criterion = ?
    #raise NotImplementedError
    # memory
    memory = ReplayMemory(capacity=args.capacity)
    # train
    train(env)
    # save model
    torch.save(behavior_net, args.model)
  # load model
  behavior_net = torch.load(args.model)
  # test
  test(env, args.render)
