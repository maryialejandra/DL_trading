import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


import itertools
import argparse
import re
import os
import pickle


from utils import ReplayBuffer 


class MLP(nn.Module):
  def __init__(self, n_inputs, n_action, n_hidden_layers=1, hidden_dim=32):
    super(MLP, self).__init__()

    M = n_inputs
    self.layers = []
    for _ in range(n_hidden_layers):
      layer = nn.Linear(M, hidden_dim)
      M = hidden_dim
      self.layers.append(layer)
      self.layers.append(nn.ReLU())

    # final layer
    self.layers.append(nn.Linear(M, n_action))
    self.layers = nn.Sequential(*self.layers)

  def forward(self, X):
    return self.layers(X)

  def save_weights(self, path):
    torch.save(self.state_dict(), path)

  def load_weights(self, path):
    self.load_state_dict(torch.load(path))



def predict(model, np_states):
  with torch.no_grad():
    inputs = torch.from_numpy(np_states.astype(np.float32))
    output = model(inputs)
    # print("output:", output)
    return output.numpy()
  


def train_one_step(model, criterion, optimizer, inputs, targets):
  # convert to tensors
  inputs = torch.from_numpy(inputs.astype(np.float32))
  targets = torch.from_numpy(targets.astype(np.float32))

  # zero the parameter gradients
  optimizer.zero_grad()

  # Forward pass
  outputs = model(inputs)
  loss = criterion(outputs, targets)
        
  # Backward and optimize
  loss.backward()
  optimizer.step()


class MultiStockEnv:
  """
  - # shares of stock 1 owned
  - price of stock 1 (using daily close price)
  - cash owned (can be used to purchase more stocks)
  Action: 
  - for each stock, you can:
  - 0 = sell
  - 1 = hold
  - 2 = buy
  """
  def __init__(self, data, initial_investment=20000):
    # data
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    # instance attributes
    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    self.action_space = np.arange(3**self.n_stock)

    # action permutations
    self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # calculate size of state
    self.state_dim = self.n_stock * 2 + 1

    self.reset()

  def reset(self):
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[self.cur_step]
    self.cash_in_hand = self.initial_investment
    return self._get_obs()

  def step(self, action):
    assert action in self.action_space

    prev_val = self._get_val()

    self.cur_step += 1
    self.stock_price = self.stock_price_history[self.cur_step]

    self._trade(action)

    cur_val = self._get_val()

    reward = cur_val - prev_val

    done = self.cur_step == self.n_step - 1

    info = {'cur_val': cur_val}

    return self._get_obs(), reward, done, info

  def _get_obs(self):
    obs = np.empty(self.state_dim)
    obs[:self.n_stock] = self.stock_owned
    obs[self.n_stock:2*self.n_stock] = self.stock_price
    obs[-1] = self.cash_in_hand
    return obs

  def _get_val(self):
      return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

  def _trade(self, action):
    action_vec = self.action_list[action]

    for i, a in enumerate(action_vec):
      if a == 0:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0
      elif a == 2:
        max_buy = self.cash_in_hand * 0.05
        buy_amount = max_buy // self.stock_price[i]
        self.stock_owned[i] += buy_amount
        self.cash_in_hand -= buy_amount * self.stock_price[i]
        # hold action is implicitly handled by not changing the state


class DQNAgent(object):
  def __init__(self, state_size, action_size, initial_capital=100000):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ReplayBuffer(state_size, action_size, size=500)
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = MLP(state_size, action_size)

    self.criterion = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters())

    self.initial_capital = initial_capital
    self.current_cash = initial_capital
    self.current_portfolio_value = 0
    self.last_buy_price = 0
    self.current_position = 0  # 0: no position, 1: holding an asset

  def update_replay_memory(self, state, action, reward, next_state, done):
    self.memory.store(state, action, reward, next_state, done)

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)
    act_values = predict(self.model, state)
    return np.argmax(act_values[0])

  def replay(self, batch_size=32):
    if self.memory.size < batch_size:
      return

    minibatch = self.memory.sample_batch(batch_size)
    states = minibatch['s']
    actions = minibatch['a']
    rewards = minibatch['r']
    next_states = minibatch['s2']
    done = minibatch['d']

    target = rewards + (1 - done) * self.gamma * np.amax(predict(self.model, next_states), axis=1)
    target_full = predict(self.model, states)
    target_full[np.arange(batch_size), actions] = target

    train_one_step(self.model, self.criterion, self.optimizer, states, target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)

def play_one_episode(agent, env, is_train, scaler, batch_size):
  state = env.reset()
  state = scaler.transform([state])
  done = False

  actions_record = []
  total_reward = 0

  while not done:
    action = agent.act(state)
    actions_record.append(action)
    next_state, reward, done, info = env.step(action)
    next_state = scaler.transform([next_state])

    reward = calculate_reward(agent, next_state, action, info)
    total_reward += reward

    if is_train == 'train':
      agent.update_replay_memory(state, action, reward, next_state, done)
      agent.replay(batch_size)

    state = next_state

  return info['cur_val'], actions_record

def calculate_reward(agent, next_state, action, info):
  if action == 2:  # Buy
    if agent.current_position == 0 and agent.current_cash > 0:
      purchase_amount = min(agent.current_cash * 0.05, agent.current_cash)
      agent.last_buy_price = info['cur_val']
      agent.current_cash -= purchase_amount
      agent.current_portfolio_value += purchase_amount
      agent.current_position = 1
    reward = (info['cur_val'] - agent.last_buy_price) / agent.last_buy_price

  elif action == 0:  # Sell
    if agent.current_position == 1:
      sell_value = agent.current_portfolio_value
      agent.current_cash += sell_value
      agent.current_portfolio_value = 0
      reward = (info['cur_val'] - agent.last_buy_price) / agent.last_buy_price
      agent.current_position = 0
    else:
      reward = -0.02  # Penalty for trying to sell without holding

  elif action == 1:  # Hold
    if agent.current_position == 1:
      reward = (info['cur_val'] - agent.last_buy_price) / agent.last_buy_price
    else:
        reward = 0

  reward = np.clip(reward, -1, 1)
  return reward
