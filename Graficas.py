import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_actions_record(filename):
    with open(filename, 'rb') as f:
        actions_record_all = pickle.load(f)
    return actions_record_all

def plot_actions(actions_record_all, num_episodes):
    actions_per_stock = np.zeros((3, num_episodes, 3))  # [stock, episode, action]
    
    for episode, actions_record in enumerate(actions_record_all):
        for action in actions_record:
            for stock in range(3):
                actions_per_stock[stock, episode, action[stock]] += 1

    actions = ['Sell', 'Hold', 'Buy']
    for stock in range(3):
        plt.figure()
        for action in range(3):
            plt.plot(actions_per_stock[stock, :, action], label=actions[action])
        plt.title(f'Stock {stock + 1} Actions over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Count of Actions')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    mode = 'train'  # or 'test' based on what you ran
    actions_record_all = load_actions_record(f'rewards_folder/actions_record_{mode}.pkl')
