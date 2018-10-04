
import numpy as np
import matplotlib.pyplot as plt

class LinearEnv:
    """
    Allowed states are:
    State id: [ 0 ... 10 ]
    Type:     [ S ...  T ]
    Reward:   [-1 -1   0 ]
    """
    def __init__(self):
        size = 9
        self.nb_st = size + 2   # nb states
        self.nb_act = 2         # nb actions
        self._max_left = 1      # last non-terminal state to the left
        self._max_right = size  # last non-terminal state to the right
        self._start_state = 0
        self.reset()

    def reset(self):
        self._state = self._start_state
        self._done = False
        return self._state

    def step(self, action):
        if self._done: return (self._state, 0, True)  # We are done
        if action not in [0, 1]: raise ValueError('Invalid action')

        self._state += -1 if action==0 else 1
        self._state = max(self._state, 0)   # bounce off the wall
        obs = self._state
        if self._state > self._max_right:
            reward = -1
            self._done = True
        elif self._state < self._max_left:
            reward = -1
            self._done = False
        else:
            reward = -1
            self._done = False
        return (obs, reward, self._done)


# Correct Q values for [0.50, 0.50] policy
REF_RANDOM = np.array([[-110.68971241, -108.685087  ],
                       [-110.66583621, -104.73818239],
                       [-108.71371977,  -98.88761006],
                       [-104.89313003,  -90.93288528],
                       [ -98.94523759,  -80.92351417],
                       [ -90.94655643,  -68.77193039],
                       [ -80.74743653,  -54.93719431],
                       [ -68.896667  ,  -38.83260026],
                       [ -54.71899209,  -20.97842456],
                       [ -38.930884  ,   -1.        ],
                       [   0.        ,    0.        ]])

# Correct Q values for [0.00, 1.00] greedy policy
REF_GREEDY = np.array([[-11., -10.],
                       [-11.,  -9.],
                       [-10.,  -8.],
                       [ -9.,  -7.],
                       [ -8.,  -6.],
                       [ -7.,  -5.],
                       [ -6.,  -4.],
                       [ -5.,  -3.],
                       [ -4.,  -2.],
                       [ -3.,  -1.],
                       [  0.,   0.]])


def generate_episode(env, policy):
    """Generate one complete episode"""
    trajectory = []
    done = True
    for _ in range(1000):  # limit episode length
        # === time step starts ===
        if done:
            St, Rt, done = env.reset(), None, False
        else:
            St, Rt, done = env.step(At)   
        At = np.random.choice([0, 1], p=policy[St])
        trajectory.append((St, Rt, done, At))
        if done:
            break
        # === time step ends here ===
    return trajectory


class LogEntry:
    """Data log for one evaluation or training run, e.g. 100 episodes"""
    def __init__(self, type_, Q_hist, perf):
        self.type = type_     # string, e.g. 'monte carlo'
        self.Q_hist = Q_hist  # history of state-values
        self.perf = perf      # agent performance (episode length)



def plot_experiments(log, truth_ref, plot_title):
    """Plot policy evaluation process
    
    Params:
        histories - list of 3d arrays, 
            each element is np.array, each np.array is independent eval or training run
            array dim 0 is number of episodes
            array dim 1 is Q-table (which is 2d on it's own) after each episode
        performances - list of 1d arrays
            each array is independent evaluation or training run
        truth_ref - precomputed correct Q-values
    """
    fig = plt.figure(figsize=[18,6])
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # plot ground truth
    ax1.plot(truth_ref[:-1,0], color='gray', linestyle='--')
    ax1.plot(truth_ref[:-1,1], color='gray', linestyle='--')
    
    for le in log:
        Q_vals = le.Q_hist[-1]
        ax1.plot(Q_vals[:-1,0], color='red', alpha=0.3)   # Q left
        ax1.plot(Q_vals[:-1,1], color='green', alpha=0.3) # Q right
        
        E_vals = np.sum(np.sum((truth_ref - le.Q_hist)**2, axis=-1), axis=-1)  # RMSE
        ax2.plot(E_vals, color='blue', alpha=0.3)         # RMSE
        
        ax3.plot(le.perf, color='blue', alpha=0.3)
        ax3.plot([0, len(le.perf)],[10,10], color='gray', linestyle='--')
    
    ax1.grid()
    ax1.set_title('Estimated State-Action Values')
    ax2.grid()
    ax2.set_title('Ground Truth Error')
    ax3.set_ylim([0, 1010])
    ax3.grid()
    ax3.set_title('Agent Performance')
    fig.suptitle(plot_title, fontsize=18)
                
    plt.show()




def make_eps_greedy(Q, eps):
    policy = np.zeros(shape=Q.shape) + eps
    maxq = np.argmax(Q, axis=-1)
    policy[range(len(policy)),maxq] = 1-eps
    return policy





def mc_control(env, policy, N, alpha, eps):
    hist, perf = [], []
    Q = np.zeros(shape=[env.nb_st, env.nb_act])

    for ep in range(N):
        trajectory = generate_episode(env, policy)
        for t in range(len(trajectory)-1):
            Gt = -(len(trajectory)-t-1)  # -1 * nb_steps_to_end; assumes disc==1
            St, _, _, At = trajectory[t]
            # Q[St, At] = Q[St, At] + (1/C[St, At]) * (Gt - Q[St, At])  # old code
            Q[St, At] = Q[St, At] + alpha * (Gt - Q[St, At])  # replace 1/C with alpha
        
        policy = make_eps_greedy(Q, eps)  # Make policy e-greedy
            
        hist.append(Q.copy())
        perf.append(len(trajectory)-1)

    return np.array(hist), np.array(perf)




def sarsa(env, policy, N, alpha, eps):
    hist, perf = [], []
    Q = np.zeros(shape=[env.nb_st, env.nb_act])
    for ep in range(N):
        trajectory = generate_episode(env, policy)
        for t in range(len(trajectory)-1):
            St, _, _, At = trajectory[t]
            St_1, Rt_1, _, At_1 = trajectory[t+1]
            target = Rt_1 + 1.0 * Q[St_1, At_1]
            Q[St, At] = Q[St, At] + alpha * (target - Q[St, At])
            
        policy = make_eps_greedy(Q, eps)  # Make policy e-greedy

        hist.append(Q.copy())
        perf.append(len(trajectory)-1)
    return np.array(hist), np.array(perf)





def calc_Gt(trajectory, Q, t, disc, nstep=float('inf')):
    """Calculates return for state t, using n future steps.
    Params:
        traj - complete trajectory, each time-step should be tuple:
            (observation, reward, done, action)
        Q (float arr) - state-action-values, Q[term_state,:] must be zero!
        t (int [t, T-1]) - calc Gt for this time step in trajectory,
            0 is initial state; T-1 is last non-terminal state
        disc - discrount, usually noted as gamma
        n (int or +inf, [1, +inf]) - n-steps of reward to accumulate
                If n >= T then calculate full return for state t
                For n == 1 this equals to TD return
                For n == +inf this equals to MC return
    """

    T = len(trajectory)-1   # terminal state
    max_j = min(t+nstep, T) # last state iterated, inclusive
    tmp_disc = 1.0          # this will decay with rate disc
    Gt = 0                  # result

    # Iterate from t+1 to t+nstep or T (inclusive on both start and finish)
    for j in range(t+1, max_j+1):
        Rj = trajectory[j][1]  # traj[j] is (obs, reward, done, action)
        Gt += tmp_disc * Rj
        tmp_disc *= disc

    # Note that Q[Sj, Aj] will have state-value of state t+nstep or
    # zero if t+nstep >= T as V[St=T] must equal 0
    Sj, _, _, Aj = trajectory[j]  # traj[j] is (obs, reward, done, action)
    Gt += tmp_disc * Q[Sj, Aj]

    return Gt



def nstep_sarsa(env, policy, N, alpha, eps, nstep):
    hist, perf = [], []
    Q = np.zeros(shape=[env.nb_st, env.nb_act])
    for ep in range(N):
        trajectory = generate_episode(env, policy)
        for t in range(len(trajectory)-1):
            St, _, _, At = trajectory[t]
            Gt = calc_Gt(trajectory, Q, t, disc=1.0, nstep=nstep)
            target = Gt
            Q[St, At] = Q[St, At] + alpha * (target - Q[St, At])
        policy = make_eps_greedy(Q, eps)
        
        hist.append(Q.copy())
        perf.append(len(trajectory))
        
    return np.array(hist), np.array(perf)



