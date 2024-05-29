import numpy as np
import itertools
from utils import ReplayBuffer

class MultiStockEnv:
    def __init__(self, data, initial_investment=20000):
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        self.action_space = np.arange(3**self.n_stock)
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

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

        action_success = self._trade(action)

        cur_val = self._get_val()
        reward = cur_val - prev_val

        if not action_success:
            reward -= 0.01  # Penalty for unsuccessful buy action

        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val, 'action_success': action_success}

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
        action_success = True

        for i, a in enumerate(action_vec):
            if a == 0:  # Sell
                ##Sell all the stocks
                if self.stock_owned[i] > 0:
                  self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                  self.stock_owned[i] = 0
                else:
                    action_success = False
            elif a == 2:  # Buy
                #Max by is the 30% of the total portfolio
                max_buy = self.cash_in_hand * 0.3
                buy_amount = max_buy // self.stock_price[i]
                ##Cannot buy if already owns a stock
                if self.stock_owned[i] > 0:
                    action_success = False
                elif buy_amount > 0:
                    self.stock_owned[i] += buy_amount
                    self.cash_in_hand -= buy_amount * self.stock_price[i]
                else:
                    action_success = False  # Failed to buy due to insufficient funds

        #print(f"Step: {self.cur_step}, Action: {action_vec}, Stocks Owned: {self.stock_owned}, Cash: {self.cash_in_hand}, Portfolio Value: {self._get_val()}")
        return action_success

import numpy as np
import torch
import torch.nn as nn

class DQNAgent:
    def __init__(self, state_size, action_size, initial_capital=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.current_portfolio_value = 0
        self.last_buy_price = 0
        self.current_position = 0

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size),np.zeros(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return (np.argmax(act_values.detach().numpy()), act_values.detach().numpy())

    def replay(self, batch_size=32):
        if self.memory.size < batch_size:
            return

        minibatch = self.memory.sample_batch(batch_size)
        states = torch.FloatTensor(minibatch['s'])
        actions = minibatch['a']
        rewards = torch.FloatTensor(minibatch['r'])
        next_states = torch.FloatTensor(minibatch['s2'])
        done = torch.FloatTensor(minibatch['d'])

        target = rewards + (1 - done) * self.gamma * torch.max(self.target_model(next_states), axis=1)[0]
        target_full = self.model(states)
        target_full[np.arange(batch_size), actions] = target

        self.optimizer.zero_grad()
        loss = self.criterion(target_full, self.model(states))
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def play_one_episode(agent, env, is_train, scaler, batch_size):
    state = env.reset()
    state = scaler.transform([state])
    done = False

    actions_record = []
    values_record = []
    q_values = []
    states = []
    total_reward = 0

    while not done:
        #print('agen_Act',agent.act(state))
        action, q_value = agent.act(state)
        actions_record.append(action)
        states.append(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])

        reward = calculate_reward(agent, next_state, action, info)
        total_reward += reward

        if is_train == 'train':
            agent.update_replay_memory(state, action, reward, next_state, done)
            agent.replay(batch_size)
            agent.update_target_model()

        state = next_state
        values_record.append(info['cur_val']) 
        q_values.append(q_value)
    return values_record, actions_record, q_values,states

def calculate_reward(agent, next_state, action, info):
    if action == 2:  # Buy
        if not info['action_success']:
            reward = -0.01 
        elif agent.current_cash > 0:
            purchase_amount = min(agent.current_cash * 0., agent.current_cash)
            agent.last_buy_price = info['cur_val']
            agent.current_cash -= purchase_amount
            agent.current_portfolio_value += purchase_amount
            agent.current_position = 1
            reward = (info['cur_val'] - agent.last_buy_price) / agent.last_buy_price
        
        #if not info['action_success']:
        #    reward -= 0.01  # Penalty for failed buy action

    elif action == 0:  # Sell
        if not info['action_success']:
            reward = -0.02 
        else:
            sell_value = agent.current_portfolio_value
            agent.current_cash += sell_value
            agent.current_portfolio_value = 0
            reward = (info['cur_val'] - agent.last_buy_price) / agent.last_buy_price
            agent.current_position = 0
        #else: 
        #   reward = -0.02  # Penalty for trying to sell without holding

    elif action == 1:  # Hold
        if agent.current_position == 1:
            reward = (info['cur_val'] - agent.last_buy_price) / agent.last_buy_price
        else:
            reward = 0  

    reward = np.clip(reward, -1, 1)
    return reward

