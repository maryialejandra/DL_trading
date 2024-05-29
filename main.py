import argparse
import re
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from Graficas import plot_actions
import matplotlib.pyplot as plt

from agent import MultiStockEnv, DQNAgent, play_one_episode
from utils import get_data, make_dir, get_scaler


def create_consolidated_table(all_episode_data, rewards_folder):
    consolidated_data = []
    for episode_index, (values, actions) in enumerate(all_episode_data):
        for iteration, (value, action) in enumerate(zip(values, actions)):
            consolidated_data.append((episode_index + 1, iteration + 1, action, value))

    df_consolidated = pd.DataFrame(consolidated_data, columns=['Episode', 'Iteration', 'Action', 'Value'])
    df_consolidated.to_csv(f'{rewards_folder}/consolidated_data.csv', index=False)
    print("Consolidated data saved to 'consolidated_data.csv'")

if __name__ == '__main__':

  # config
  models_folder = 'rl_trader_models'
  rewards_folder = 'rl_trader_rewards'
  num_episodes = 10
  batch_size = 32
  initial_investment = 20000


  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  args = parser.parse_args()

  make_dir(models_folder)
  make_dir(rewards_folder)

  data = get_data('EURUSD.csv')
  #print(data.shape)
  if len(data.shape) == 1:
    data = data.reshape((-1,1))
  n_timesteps, n_stocks = data.shape

  n_train = n_timesteps // 2

  train_data = data[:n_train]
  test_data = data[n_train:]

  env = MultiStockEnv(train_data, initial_investment)
  state_size = env.state_dim
  action_size = len(env.action_space)
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  # store the final value of the portfolio (end of episode)
  

  if args.mode == 'test':
    # then load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # remake the env with test data
    env = MultiStockEnv(test_data, initial_investment)

    # make sure epsilon is not 1!
    # no need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon = 0.01

    # load trained weights
    agent.load(f'{models_folder}/dqn.ckpt')

  
    val, actions_record,q_values,states = play_one_episode(agent, env, args.mode,scaler,batch_size)
    print(len(test_data))
    print(len(actions_record))
    color_dict = {
      0:'r',
      1:'w',
      2: 'g'
     }
    #print(val)
    for i in range(len(test_data)):
      if i == 0:
        plt.plot(i,test_data[i],color=color_dict[1],marker='o')
      else:
        plt.plot(i,test_data[i],color=color_dict[actions_record[i-1]],marker='o')
    plt.show()
      
    
  
  # play the game num_episodes times
  all_episode_data = []
  portfolio_value = []
  actions_record_all = []
  all_states = []
  all_q_values = []
  for e in range(num_episodes):
    t0 = datetime.now()
    values_record, actions_record,q_values,states = play_one_episode(agent, env, args.mode,scaler,batch_size)
    dt = datetime.now() - t0
    print(f"episode: {e + 1}/{num_episodes}, episode end value: {values_record[-1]:.2f}, duration: {dt}")
    portfolio_value.append(values_record[-1])
    actions_record_all.append(actions_record)
    all_episode_data.append((values_record, actions_record))
    all_states.extend(states)
    all_q_values.extend(q_values)
  
  # Create and save Q-table
  print("States collected:", len(all_states))
  print("Q-values collected:", len(all_q_values))
  q_table_data = [(state, action, q_value) for state, q_values in zip(all_states, all_q_values) for action, q_value in enumerate(q_values)]
  print(q_table_data)
  df_q_table = pd.DataFrame(q_table_data, columns=['State', 'Action', 'Q_Value'])
  df_q_table.to_csv(f'{rewards_folder}/q_table.csv', index=False)
  print("Q-table saved to 'q_table.csv'")
  
  # save the weights when we are done
  if args.mode == 'train':
    # save the DQN
    agent.save(f'{models_folder}/dqn.ckpt')

    # save the scaler
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)

  create_consolidated_table(all_episode_data, rewards_folder)
  # save portfolio value for each episode
  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
  with open(f'{rewards_folder}/actions_record_{args.mode}.pkl', 'wb') as f:
    pickle.dump(actions_record_all, f)

  plot_actions(actions_record_all, len(actions_record_all))