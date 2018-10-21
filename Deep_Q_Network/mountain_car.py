import numpy as np
import matplotlib.pyplot as plt

class MountainCarEnv:
    """Mountain Car as described in Sutton & Barto (2018), example 10.1"""
    def __init__(self):
        self._pos = 0
        self._vel = 0
        self.state_low = np.array([-1.2, -0.07])   # min pos, min vel
        self.state_high = np.array([0.5, 0.07])    # max pos, max vel
        self.act_space = np.array([0, 1, 2])       # [left, idle, right]
        self.reset()

    def reset(self):
        self._pos = np.random.uniform(-0.6, -0.4)  # start pos
        self._vel = 0.0
        self._done = False
        return np.array([self._pos, self._vel], dtype=float)  # use np arrays everywhere

    def step(self, action):
        assert self._done == False
        assert action in self.act_space

        self._vel = self._vel + 0.001*(action-1) - 0.0025*np.cos(3*self._pos)
        self._vel = min(max(self._vel, -0.07), 0.07)

        self._pos = self._pos + self._vel
        self._pos = min(max(self._pos, -1.2), 0.5)

        if self._pos == -1.2:
            self._vel = 0.0

        if self._pos == 0.5:
            obs = np.array([self._pos, self._vel], dtype=float)
            reward = -1
            self._done = True
            return obs, reward, self._done
        else:
            obs = np.array([self._pos, self._vel], dtype=float)
            reward = -1
            return obs, reward, self._done