
import numpy as np

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.agent_position = [size-1, 0]
        self.goal_position = [0, size-1]

    def reset(self):
        self.agent_position = [self.size-1, 0]
        return self.agent_position

    def step(self, action):
        # Action: 0=up, 1=down, 2=left, 3=right
        if action == 0 and self.agent_position[0] > 0:
            self.agent_position[0] -= 1
        if action == 1 and self.agent_position[0] < self.size - 1:
            self.agent_position[0] += 1
        if action == 2 and self.agent_position[1] > 0:
            self.agent_position[1] -= 1
        if action == 3 and self.agent_position[1] < self.size - 1:
            self.agent_position[1] += 1

        # Check if the goal is reached
        if self.agent_position == self.goal_position:
            return self.agent_position, 1, True
        return self.agent_position, -1, False


class SimpleRLAgent:
    def __init__(self, grid_size=4, use_sorl=False):
        self.grid_size = grid_size
        self.use_sorl = use_sorl
        self.Q = np.zeros((grid_size, grid_size, 4))
        self.epsilon = 0.9
        self.alpha = 0.1
        self.gamma = 0.9

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)
        return np.argmax(self.Q[state[0], state[1], :])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state[0], next_state[1], :])
        td_target = reward + self.gamma * self.Q[next_state[0], next_state[1], best_next_action]
        td_error = td_target - self.Q[state[0], state[1], action]
        
        # For SORL, introduce a computational penalty
        if self.use_sorl:
            penalty = np.random.uniform(0, 0.05)
            td_error -= penalty
        
        self.Q[state[0], state[1], action] += self.alpha * td_error

    def train(self, env, episodes=500):
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
            # Decay epsilon
            self.epsilon *= 0.995

