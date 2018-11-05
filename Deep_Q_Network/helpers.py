import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import ListedColormap

import tiles3           # by Richard Sutton, http://incompleteideas.net/tiles/tiles3.html

from collections import namedtuple
ObsSpace = namedtuple('ObsSpace', 'shape low high')

def running_mean(x, n):
    # res = []
    # for i in range(len(x)):
    #     res.append( sum(x[max(i-n+1, 0): i+1])   /   min(i+1, n) )
        
    return [sum(x[max(i-n+1, 0): i+1])   /   min(i+1, n) for i in range(len(x))]
        
    return res


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



def plot_mountain_car(env, episode, total_tstep, steps_to_plot, trace, mem,
    axis_labels, action_labels, action_colors):
    
    q_arr = trace.q_values[total_tstep]
    states = trace.states[-steps_to_plot:]
    actions = trace.actions[-steps_to_plot:]
    
    epsilons = trace.epsilons
    
    fig = plt.figure(figsize=[12,8])

    if q_arr is not None:
        ax = fig.add_subplot(231, projection='3d')
        plot_q_max_3d(q_arr, env, title='Q_Max', 
                      labels=axis_labels, alpha=.4, axis=ax)

    ax = fig.add_subplot(232)
    plot_trajectory(states, actions, env, title='Trajectory', labels=axis_labels, axis=ax)
    
    
    if q_arr is not None:
        ax = fig.add_subplot(233)
        xx = trace.q_values.keys()
        values_tmp = np.array(list(trace.q_values.values()))  # shape [n, 128, 128, act]
        values = values_tmp[:,values_tmp.shape[1]//2, values_tmp.shape[2]//2,:]
        for i in range(values.shape[-1]):
            ax.plot(values[:,i], color=action_colors[i])
        ax.grid()
    
    if q_arr is not None:
        ax = fig.add_subplot(234)
        plot_policy(q_arr, env, labels=axis_labels,
                    colors=action_colors, collab=action_labels, axis=ax)
        
    ax = fig.add_subplot(235)
    st, act, rew_1, st_1, dones_1, _ = mem.pick_last(len(mem))
    plot_trajectory(st, act, env, title='Memory Buffer', labels=axis_labels, alpha=0.1, axis=ax)
    
    ax = fig.add_subplot(236)
    plot_episode_rewards(trace.ep_end_idx, trace.ep_rewards, ax)

    plt.tight_layout()
    plt.show()



def plot_episode_rewards(ep_end_dict, rew_dict, axis=None):
    
    tsteps = []    # episodes end tsteps
    rewards = []   # episodes rewards
    
    for ep, end_tstep in ep_end_dict.items():
        tsteps.append(end_tstep)
        rewards.append(rew_dict[ep])
    
    rewards_avg = running_mean(rewards, 100)
    
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)
    
    if len(tsteps) > 0:
        axis.scatter(tsteps, rewards, alpha=0.5, s=1)
        axis.plot(tsteps, rewards_avg, alpha=1, color='orange')
        
    axis.grid()
    axis.set_xlabel('Time Step')
    axis.set_ylabel('Episode Reward')
    axis.set_title('Episode Rewards')




def plot_trajectory(states, actions, env, title, labels, alpha=1.0, axis=None):
    if not isinstance(states, np.ndarray): states = np.array(states)
    if not isinstance(actions, np.ndarray): actions = np.array(actions)
    
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)
    
    if len(states) == 0:
        axis.scatter(np.array([]), np.array([]))
    else:
        axis.scatter(states[actions==0,0], states[actions==0,1], marker='.', s=1, color='red', alpha=alpha)
        axis.scatter(states[actions==1,0], states[actions==1,1], marker='.', s=1, color='blue', alpha=alpha)
        axis.scatter(states[actions==2,0], states[actions==2,1], marker='.', s=1, color='green', alpha=alpha)
        
    x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
    y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]
    axis.set_xticks([x_min, x_max])
    #axis.set_xticklabels([x_min,x_max])
    axis.set_yticks([y_min, y_max])
    #axis.set_yticklabels([y_min,y_max])
    
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)




def plot_policy(q_arr, env, labels, colors, collab, axis=None):
    q_pol = np.argmax(q_arr, axis=-1)
    
    cmap = ListedColormap(colors)
    
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)
        
    heatmap = axis.pcolormesh(q_pol.T, cmap=cmap)
    axis.set_aspect('equal', 'datalim')
    cbar = plt.colorbar(heatmap)
    cbar.set_ticks(range(len(collab)))
    cbar.set_ticklabels(collab)
    
    x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
    y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]
    axis.set_xticks([0, q_arr.shape[0]])
    axis.set_xticklabels([x_min,x_max])
    axis.set_yticks([0, q_arr.shape[1]])
    axis.set_yticklabels([y_min,y_max])
    
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title('Policy')



def plot_q_max_3d(q_arr, env, color='#1f77b4', alpha=1.,
                  title='', labels=['x','y'], axis=None):
    """Plot 3D wireframe
    
    Params:
        q_arr     - 2d array with dim: [state_x, state_y]
        env       - environment with members:
                      st_low - state space low boundry e.g. [-1.2, -0.07]
                      st_high - state space high boundry
        color     - plot color
        alpha     - plot transparency
        labels    - string array [label_x, label_y, label_z], len=3, empty str to omit
        axis      - axis to plot to, if None create new figure
    """
    q_max = np.max(q_arr, axis=-1)  # calc max and inverse
    
    x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
    y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]
    x_space = np.linspace(x_min, x_max, num=q_max.shape[0])
    y_space = np.linspace(y_min, y_max, num=q_max.shape[1])
    Y, X = np.meshgrid(y_space, x_space)
    
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')

    axis.plot_wireframe(X, Y, q_max, color=color, alpha=alpha)
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_xticks([x_min, x_max])
    axis.set_yticks([y_min, y_max])
    axis.set_title(title)
    
    axis.view_init(40, -70)


# old version
# def eval_state_action_space(model, env, split=[32,32]):
#     """Evaluate 2d Q-function on area and return as 3d array
    
#     Params:
#         model     - function approximator with method: model.eval(state, action) -> float
#         env       - environment with members:
#                       st_low - state space low boundry e.g. [-1.2, -0.07]
#                       st_high - state space high boundry
#                       act_space - action space, e.g. [0, 1, 2]
#         split     - number of data points in each dimensions, e.g. [20, 20]
#     """
#     x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
#     y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]
#     x_split, y_split = split
    
#     q_arr = np.zeros([x_split, y_split, env.action_space.n])

#     for pi, pos in enumerate(np.linspace(x_min, x_max, x_split)):
#         for vi, vel in enumerate(np.linspace(y_min, y_max, y_split)):
#             q_values = model.eval(states=np.array([[pos, vel]]))[0]
#             for act in range(env.action_space.n):
#                 q_arr[pi, vi, act] = q_values[act]
                
#     return q_arr


def eval_state_action_space(model, env, split=[32,32]):
    """Evaluate 2d Q-function on area and return as 3d array
    
    Params:
        model     - function approximator with method: model.eval(state, action) -> float
        env       - environment with members:
                      st_low - state space low boundry e.g. [-1.2, -0.07]
                      st_high - state space high boundry
                      act_space - action space, e.g. [0, 1, 2]
        split     - number of data points in each dimensions, e.g. [20, 20]
    """
    # prep states to evaluate
    x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
    y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]
    x_split, y_split = split
    x_space = np.linspace(x_min, x_max, x_split)
    y_space = np.linspace(y_min, y_max, y_split)
    Y, X = np.meshgrid(y_space, x_space)
    states = np.stack([X, Y], axis=-1)
    states = states.reshape([-1, 2])
    
    q_arr = model.eval(states=states)
    return q_arr.reshape([128,128,-1])








class TileCodingFuncApprox():
    def __init__(self, st_low, st_high, nb_actions, learn_rate, num_tilings, init_val):
        """
        Params:
            st_low       - state space low boundry, e.g. [-1.2, -0.07] for mountain car
            st_high      - state space high boundry in all dimensions
            action_space - list of possible actions
            learn_rate   - step size, will be adjusted for nb_tilings automatically
            num_tilings  - tiling layers - should be power of 2 and at least 4*len(st_low)
            init_val     - initial state-action values
        """
        st_low = np.array(st_low);    st_high = np.array(st_high)
        assert len(st_low) == len(st_high)
        self._n_dim = len(st_low)
        self._nb_actions = nb_actions
        self._lr = learn_rate / num_tilings
        self._num_tilings = num_tilings
        self._scales = self._num_tilings / (st_high - st_low)
        
        # e.g. 8 tilings, 2d space, 3 actions
        # nb_total_tiles = (8+1) * (8+1) * 8 * 3
        nb_total_tiles = (num_tilings+1)**self._n_dim * num_tilings * nb_actions
                
        self._iht = tiles3.IHT(nb_total_tiles)
        self._weights = np.zeros(nb_total_tiles) + init_val / num_tilings
        
    def eval(self, states):
        assert isinstance(states, np.ndarray)
        assert states.ndim == 2
        
        all_q_values = []
        for state in states:
            assert len(state) == self._n_dim
            scaled_state = np.multiply(self._scales, state)  # scale state to map to tiles correctly
            q_values = []
            for action in range(self._nb_actions):
                active_tiles = tiles3.tiles(                 # find active tiles
                    self._iht, self._num_tilings,
                    scaled_state, [action])
                q_val = np.sum(self._weights[active_tiles])  # pick correct weights and sum up
                q_values.append(q_val)                       # store result for this action
            all_q_values.append(q_values)
        return np.array(all_q_values)

    def train(self, states, actions, targets):
        assert isinstance(states, np.ndarray)
        assert isinstance(actions, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert states.ndim == 2
        assert actions.ndim == 1
        assert targets.ndim == 1
        assert len(states) == len(actions) == len(targets)
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            target = targets[i]
            
            assert len(state) == self._n_dim
            assert np.isscalar(action)
            assert np.isscalar(target)
            
            scaled_state = np.multiply(self._scales, state)  # scale state to map to tiles correctly
            active_tiles = tiles3.tiles(                     # find active tiles
                self._iht, self._num_tilings,
                scaled_state, [action])
            value = np.sum(self._weights[active_tiles])      # q-value for state-action pair
            delta = self._lr * (target - value)              # grad is [0,1,0,0,..]
            self._weights[active_tiles] += delta             # ..so we pick active weights instead