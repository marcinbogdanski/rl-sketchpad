import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_cliffwalk(Q=None, path_blue=None, path_red=None, labels=None, saveimg=None):
    """Plot windy gridworld.
    
    Will plot optimal action for each cell, according to 'Q'
    and blue/red lines following 'path_blue'/'path_red' parameters
    
    Params:
        Q         - optional, dictionary mappting [(x,y),action] -> float
        path_blue - optional, list [(x1,y1), (x2,y2), ...] of agent steps
        path_red  - optional, list [(x1,y1), (x2,y2), ...] of agent steps
        labels    - tuple of 2x strings, e.g. ('Sarsa', 'Q-Learning')
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlim(-.5, 11.5)   # cliffwalk x range [0..11]
    ax.set_ylim(-.5, 3.5)    #           y range [0..3]
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')
    ax.set_aspect('equal', 'datalim')

    # Start, Goal, The Cliff markers
    params = {'horizontalalignment':'center', 'verticalalignment':'center'}
    ax.text(0, 0, 'S', fontsize=18, horizontalalignment='center', verticalalignment='center')
    ax.text(11, 0, 'G', fontsize=18, horizontalalignment='center', verticalalignment='center')
    ax.text(5.5, 0, 'T h e   C l i f f', fontsize=18, **params)
    
    for x in range(12):
        for y in range(4):
            if x not in [0, 11] and y == 0:  # cliff
                ax.add_patch(patches.Rectangle([x-0.5, y-0.5], 1, 1, fill=True, color='lightgray'))    
            else:
                ax.add_patch(patches.Rectangle([x-0.5, y-0.5], 1, 1, fill=False))  # cells

            if Q is not None:
                params = {'head_width':0.2, 'head_length':0.2, 'color':'gray', 'alpha':.2}
                A_star = np.argmax([Q[(x,y),a] for a in [0, 1, 2, 3]])  # pick best action
                if A_star == 3:    ax.arrow(x, y, 0,  .1, **params)  # up
                elif A_star == 1:  ax.arrow(x, y, 0, -.1, **params)  # down
                elif A_star == 0:  ax.arrow(x, y, -.1, 0, **params)  # left
                elif A_star == 2:  ax.arrow(x, y,  .1, 0, **params)  # right

    # plot blue line
    if path_blue is not None:
        for i in range(len(path_blue)-1):
            x, y = path_blue[i]
            x_, y_ = path_blue[i+1]
            if labels is not None and i == 0:
                ax.plot([x, x_], [y, y_], color='blue', alpha=1., label=labels[0])
            else:
                ax.plot([x, x_], [y, y_], color='blue', alpha=1.)

    # plot red line
    if path_red is not None:
        for i in range(len(path_red)-1):
            x, y = path_red[i]
            x_, y_ = path_red[i+1]
            if labels is not None and i == 0:
                ax.plot([x, x_], [y, y_], color='red', alpha=1., label=labels[1])
            else:
                ax.plot([x, x_], [y, y_], color='red', alpha=1.)

    fig.legend(loc=4)
                
    plt.tight_layout()
    if saveimg is not None:
        plt.savefig(saveimg)
    plt.show()