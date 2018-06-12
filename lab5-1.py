import gym

import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

action_space_n = 4
state_space_n = 16

# Initialize table with all zeros
Q = np.zeros([state_space_n, action_space_n])

# Set learning parameters
learning_rate = .85
dis = .99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-table learning algorithm
    while not done:
        # Choose an action by greedily (with noise) picking from Q-table
        action = np.argmax(Q[state, :] + np.random.randn(1, action_space_n) / (i + 1))

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-table with new knowledge using learning rate
        Q[state, action] = (1 - learning_rate) * Q[state, action] \
            + learning_rate * (reward + dis * np.max(Q[new_state, :]))

        rAll += reward
        state = new_state

    rList.append(rAll)

env.close()

print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
