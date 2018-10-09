import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plot_blackjack(Q, pi, save_path=None):
    """Plot policy and state-values.
    
    Params:
        Q - dictionary with keys in following format:
            ( (player_sum, dealer_card, has_ace), action )  ->  Q-Value
            ( (    int   ,      int   ,   bool ),   int )
    """
    def plot_policy_helper(ax, arr):
        assert arr.shape == (10, 10)
        ax.imshow(arr, origin='lower', vmin=-1, vmax=1, cmap='RdYlGn', alpha=0.3,
              extent=[0.5,10.5,11.5,21.5], interpolation='none');
        ax.set_xticks(np.arange(1,11, 1));
        ax.set_yticks(np.arange(12,22, 1));

    def plot_3d_helper(ax, Z):
        assert Z.shape == (10, 10)
        dealer_card = list(range(1, 11))
        player_points = list(range(12, 22))
        X, Y = np.meshgrid(dealer_card, player_points)
        ax.plot_wireframe(X, Y, Z)
    
    pi_no_ace = np.zeros([10,10])
    pi_has_ace = np.zeros([10,10])
    for ps in range(12, 22):     # convert player sum from 12-21 to 0-9
        for dc in range(1, 11):  # convert dealer card from 1-10 to 0-9
            # store -1 if no data, this should almost never happen
            pi_no_ace[ps-12, dc-1] =  pi[(ps,dc,False)]
            pi_has_ace[ps-12, dc-1] = pi[(ps,dc,True)]
            
    V_has_ace = np.zeros([10,10])
    V_no_ace = np.zeros([10,10])
    for ps in range(12, 22):     # convert player sum from 12-21 to 0-9
        for dc in range(1, 11):  # convert dealer card from 1-10 to 0-9
            # store -1 if no data, this should almost never happen
            V_no_ace[ps-12, dc-1] = Q[((ps,dc,False), pi[(ps,dc,False)])]
            V_has_ace[ps-12, dc-1] = Q[((ps,dc,True), pi[(ps,dc,True)])]
    
    fig = plt.figure(figsize=[8,6])
    fig.text(0., 0.70, 'Usable\n  Ace', fontsize=12)
    fig.text(0., 0.25, '   No\nUsable\n  Ace', fontsize=12)
    
    # Policy, with ace
    ax = plt.subplot2grid([2,3], [0, 0], fig=fig)
    plot_policy_helper(ax, pi_has_ace)
    ax.set_title(r'$\pi_*$', fontsize=24)
    ax.text(8, 20, 'Stick')
    ax.text(8, 13, 'HIT')
    
    # Policy, no ace
    ax = plt.subplot2grid([2,3], [1, 0], fig=fig)
    plot_policy_helper(ax, pi_no_ace)
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.yaxis.set_label_position('right')
    
    # 3d, with ace
    ax = plt.subplot2grid([2,3], [0, 1], colspan=2, fig=fig, projection='3d')
    plot_3d_helper(ax, V_has_ace)
    ax.set_title(r'$v_*$', fontsize=24)
    ax.set_zticks([])
    
    # 3d, no ace
    ax = plt.subplot2grid([2,3], [1, 1], colspan=2, fig=fig, projection='3d')
    plot_3d_helper(ax, V_no_ace)
    ax.set_xlabel('Dealer Showing');
    ax.set_ylabel('Player Sum')
    ax.set_zticks([])
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()

    
def plot_Q_trace(trace, has_ace, start_at=0, full_scale=False):
    assert has_ace in [0, 1]
    np_trace = np.array(trace)

    fig = plt.figure(figsize=[12,12])
    for ph in range(10):
        for ds in range(10):
            ax=fig.add_subplot(10,10,(((9-ph)*10)+ds)+1)
            #               TR,  ph,ds,ha,  act
            ax.plot( np_trace[start_at:, ph,ds,has_ace, 0], color='blue', alpha=1. )
            ax.plot( np_trace[start_at:, ph,ds,has_ace, 1], color='red', alpha=1. )
            
            ax.set_yticks([]); ax.set_xticks([])
            
            if full_scale:
                ax.set_ylim([-1, 1])

            if ph == 0:
                ax.set_xlabel(str(ds+1))
            if ds == 0:
                ax.set_ylabel(str(ph+12))
                

    if has_ace:
        print('Q trace - has ace - start at:', start_at, '- full scale:', full_scale)
    else:
        print('Q trace - no ace - start at:', start_at, '- full scale:', full_scale)
    fig.text(0, 0.5, 'Player Sum', rotation='vertical')
    fig.text(0.45, 0, 'Dealer Shows')

    plt.tight_layout()
    plt.show()