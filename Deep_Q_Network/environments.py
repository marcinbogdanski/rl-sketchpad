import numpy as np
import gym

from collections import namedtuple
ObsSpace = namedtuple('ObsSpace', 'shape low high')

class BoxSpace():
    def __init__(self, shape, low, high):
        assert isinstance(shape, tuple)
        assert isinstance(low, list)
        assert isinstance(high, list)
        self.shape = shape
        self.low = np.array(low)
        self.high = np.array(high)
        
class DiscreteSpace():
    def __init__(self, n):
        assert isinstance(n, int)
        self.n = n
    def sample(self):
        return np.random.randint(self.n)
    

class MountainCarEnv:
    """Mountain Car as described in Sutton & Barto (2018), example 10.1"""
    def __init__(self):
        
        self.observation_space = BoxSpace(
            shape=(2,), low=[-1.2, -0.07], high=[0.5, 0.07])
        
        self.action_space = DiscreteSpace(n=3)
        
        self._pos = 0
        self._vel = 0
        self.reset()

    def reset(self):
        self._pos = np.random.uniform(-0.6, -0.4)  # start pos
        self._vel = 0.0
        self._done = False
        return np.array([self._pos, self._vel], dtype=float)  # use np arrays everywhere

    def step(self, action):
        assert self._done == False
        assert 0 <= action < self.action_space.n

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
            return obs, reward, self._done, None
        else:
            obs = np.array([self._pos, self._vel], dtype=float)
            reward = -1
            return obs, reward, self._done, None


class Pendulum2DEnv():
    def __init__(self):
        self._env = gym.make('Pendulum-v0')
        self.observation_space = BoxSpace(
            shape=(2,), low=[-np.pi, -8.0], high=[np.pi, 8.0]
        )
        self.action_space = DiscreteSpace(n=3)
        
    def reset(self):
        cos, sin, vel = self._env.reset()
        theta = np.arctan2(sin, cos)
        return np.array([theta, vel])
        
    def step(self, action):
        torques = [-2.0, 0.0, 2.0]
        # torques = [-2.0, -.5, 0.0, .5, 2.0]
        joint_effort = torques[action]
        
        obs, rew, done, _ = self._env.step([joint_effort])
        cos, sin, vel = obs
        theta = np.arctan2(sin, cos)
        return np.array([theta, vel]), rew, done, obs
    
    def render(self):
        self._env.render()
        
    def close(self):
        self._env.close()


class CartPole2DEnv():
    def __init__(self):
        self._env = gym.make('CartPole-v0')
        self.observation_space = BoxSpace(
            shape=(2,), low=[-.5, -4.0], high=[+.5, +4.0]
        )
        self.action_space = DiscreteSpace(n=2)
        
    def reset(self):
        cart_pos, cart_vel, pole_angle, pole_vel = self._env.reset()
        return np.array([pole_angle, pole_vel])
    
    def step(self, action):
        obs, rew, done, _ = self._env.step(action)
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        return np.array([pole_angle, pole_vel]), rew, done, obs
    
    def render(self):
        self._env.render()
        
    def close(self):
        self._env.close()
