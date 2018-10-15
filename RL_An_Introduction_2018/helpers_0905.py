import numpy as np
import matplotlib.pyplot as plt

class LinearEnv:
    """
    State nb:   [   0       1   ...   499   500   501   ...   1000    1001  ]
    State type: [terminal   .   ...    .   start   .    ...     .   terminal]
    """
    V_approx = np.arange(-1001, 1001, 2) / 1000.0  # ignore nonlinearity at terminal states
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self._state = 500
        self._done = False
        return self._state
        
    def step(self, action):
        if self._done: raise ValueError('Episode has terminated')
    
        if action == 0:   self._state -= np.random.randint(1, 101)  # step left
        elif action == 1: self._state += np.random.randint(1, 101)  # step right
        else: raise ValueError('Invalid action')
            
        self._state = np.clip(self._state, 0, 1001)  # clip to 0..1001
        if self._state in [0, 1001]:                 # both 0 and 1001 are terminal
            self._done = True
        
        if self._state == 0:
            return self._state, -1, self._done       # state, rew, done
        elif self._state == 1001:
            return self._state, 1, self._done
        else:
            return self._state, 0, self._done

def plot_linear(V, env, freq=None, saveimg=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Note, state 0 is terminal, so we exclude it
    ax.plot(range(1,1001), env.V_approx[1:], color='red', linewidth=0.8, label='"True" value')
    ax.plot(range(1,1001), V[1:], color='blue', linewidth=0.8, label='Approx. MC value')
    ax.set_xlabel('State')
    ax.set_ylabel('Value Scale')
    ax.legend()
    
    if freq is not None:
        mu = np.zeros(len(V))       # convert to array, V has same len as freq
        for st in range(len(V)):
            mu[st] = freq[st]
        ax2 = ax.twinx()
        # again, exclude terminal state 0
        ax2.bar(range(1,1001), mu[1:], color='gray', width=5, label='State Distr.')
        ax2.set_ylabel('Distribution Scale')
        ax2.legend(loc='right', bbox_to_anchor=(1.0, 0.2))
    
    plt.tight_layout()
    if saveimg is not None:
        plt.savefig(saveimg)
    plt.show()   