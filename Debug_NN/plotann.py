import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d



def show_neurons_weights(weights_iwn, gradients_iwn, neurons, title_prefix='', mode='median', color='black', figsize=None):
    assert weights_iwn.ndim == 3
    assert gradients_iwn.ndim == 3
    assert weights_iwn.shape == gradients_iwn.shape
    
    for n in neurons:
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=figsize)
        plot_weights(weights_iwn, neuron_nb=n, title=title_prefix+' weights #'+str(n), axis=ax1)
        plot_update_ratios(weights_iwn, neuron_nb=n, title=title_prefix+' ratios #'+str(n), mode=mode, color=color, axis=ax2)
        plot_gradients(gradients_iwn, neuron_nb=n, title=title_prefix+' gradients #'+str(n), mode=mode, color=color, axis=ax3)
        fig.tight_layout()
        
def show_biases(biases_in, gradients_in, title_prefix='', mode='median', color='black', figsize=None):
    assert biases_in.ndim == 2
    assert gradients_in.ndim == 2
    assert biases_in.shape == gradients_in.shape
    
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=figsize)
    plot_weights(biases_in[:,:,np.newaxis], neuron_nb=0, title=title_prefix+' biases', axis=ax1)
    plot_update_ratios(biases_in[:,:,np.newaxis], neuron_nb=0, title=title_prefix+' ratios', mode=mode, color=color, axis=ax2)
    plot_gradients(gradients_in[:,:,np.newaxis], neuron_nb=0, title=title_prefix+' gradients', mode=mode, color=color, axis=ax3)
    fig.tight_layout()

def show_layer_summary(weights_iwn, gradients_iwn, title_prefix='', mode='median', color='black', figsize=None):
    assert weights_iwn.ndim == 3
    assert gradients_iwn.ndim == 3
    assert weights_iwn.shape == gradients_iwn.shape
    
    # Hidden 1: Summary
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plot_update_ratios(weights_iwn, neuron_nb='all', title=title_prefix+' ratios', mode=mode, color=color, axis=ax1)
    plot_gradients(gradients_iwn, neuron_nb='all', title=title_prefix+' gradients', mode=mode, color=color, axis=ax2)
    fig.tight_layout()
    
    return fig


    
def show_neurons_activations(activations_ian, epoch_size, activation_function, neurons, title_prefix='', color=(0,0,0,1)):
    assert activations_ian.ndim == 3
    
    skip_first = False
    if activation_function in ['relu', 'lrelu']:
        skip_first = True
        
    nb_neurons = len(neurons)
    nb_columns = 4
    nb_rows = (nb_neurons-1)//2 + 2
    
    width = 16
    height = 2 * nb_rows
    fig = plt.figure(figsize=[width, height])
    
    colorB = (color[0]/2, color[1]/2, color[2]/2, color[3])  # div by 2, but not alpha
        
    nb_plot = 1
    for neuron_nb in neurons:
        ax = fig.add_subplot(nb_rows, nb_columns, nb_plot, projection='3d')
        plot_3d_histogram(activations_ian, es=epoch_size, neuron_nb=neuron_nb, 
                          title=title_prefix+' Neuron #'+str(neuron_nb)+ 'raw', funct=None,
                          color=colorB, ax=ax)
        nb_plot += 1
        
        ax = fig.add_subplot(nb_rows, nb_columns, nb_plot, projection='3d')
        plot_3d_histogram(activations_ian, es=epoch_size, neuron_nb=neuron_nb, 
                          title=title_prefix+' Neuron #'+str(neuron_nb), funct=activation_function,
                          skip_first=skip_first, color=color, ax=ax)
        nb_plot += 1

    fig.tight_layout()
    
    
def show_layer_activations(activations_ian, epoch_size, activation_function, title_prefix='',
                           color=(0,0,0,1), lines_01=True, figsize=None):
    assert activations_ian.ndim == 3
    #assert len(activations_ian) % epoch_size == 0
    
    skip_first = False
    if activation_function in ['relu', 'lrelu']:
        skip_first = True
        
    colorB = (color[0]/2, color[1]/2, color[2]/2, color[3])  # div by 2, but not alpha
    
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, subplot_kw=dict(projection='3d'), figsize=figsize)
    plot_3d_histogram(activations_ian, es=epoch_size, neuron_nb='all', title=title_prefix+' raw',
                      funct=None, color=colorB, lines_01=lines_01, ax=ax1)
    plot_3d_histogram(activations_ian, es=epoch_size, neuron_nb='all', title=title_prefix,
                      skip_first=skip_first, funct=activation_function, color=color, lines_01=lines_01, ax=ax2)
    fig.tight_layout()
    
    
    
def plot_weights(data_iwn, neuron_nb, title=None, axis=None):
    assert data_iwn.ndim == 3
    assert np.isscalar(neuron_nb)
    
    ni, nw, nn = data_iwn.shape   # iter, weights
    
    if axis is None:
        fig, axis = plt.subplots(1,1, figsize=[16,6])
    
    axis.plot(data_iwn[:,:,neuron_nb], alpha=max(100/nw, 0.002))
    
    if title is not None:
        axis.set_title(title + ' # ' + str(neuron_nb))
        
        
        
def plot_gradients(data, neuron_nb, title=None, mode='median', color='black', axis=None, figsize=None):
    assert data.ndim == 3
    
    ni, nw, nn = data.shape  # iter, weights, neurons
    
    if neuron_nb == 'all':
        data_iw = data.reshape([ni, -1])
    elif np.issubdtype(type(neuron_nb), np.integer):
        data_iw = data[:,:,neuron_nb]
    else:
        raise ValueError('neuron_nb must be int or "all"')
        
    ratio_abs = np.abs(data_iw)
        
    if axis is None:
        fig, axis = plt.subplots(1,1, figsize=figsize)
        
    if mode == 'full':
        axis.plot(ratio_abs, alpha=max(1/nw, 0.002), color='pink');

    if mode == 'median':
        axis.plot(np.min(ratio_abs, axis=-1), alpha=.05, color=color);
        axis.plot(np.max(ratio_abs, axis=-1), alpha=.05, color=color);
        axis.plot(np.percentile(ratio_abs, 10, axis=-1), alpha=.3, color=color);
        axis.plot(np.percentile(ratio_abs, 90, axis=-1), alpha=.3, color=color);
        axis.plot(np.median(ratio_abs, axis=-1), alpha=1, color=color);
        
    if mode == 'norms':
        data_norm_i = np.linalg.norm(data_iw, axis=-1)
        axis.plot(data_norm_i, alpha=1, color=color)
        #data_delta_norm_i = np.linalg.norm(data_delta_iw, axis=-1)
        #axis.plot(data_delta_norm_i/data_norm_i[:-1], alpha=1, color='green')
        
    if mode == 'dl4j':
        tmp_ = np.mean(np.abs(data_iw), axis=-1)
        axis.plot(tmp_, color=color)
    
    axis.plot([0,len(ratio_abs)],[.1, .1], ls='--', c='black')
    axis.plot([0,len(ratio_abs)],[.01, .01], ls='-', c='black')
    axis.plot([0,len(ratio_abs)],[.001, .001], ls='--', c='black')
    axis.set_yscale('log')

    if title is not None:
        axis.set_title(title + ' ' + mode)
        
        
        
def plot_update_ratios(data, neuron_nb, title=None, mode='median', color='red', axis=None, figsize=None):
    assert data.ndim == 3
    
    ni, nw, nn = data.shape  # iter, weights, neurons
    
    if neuron_nb == 'all':
        data_iw = data.reshape([ni, -1])
    elif np.issubdtype(type(neuron_nb), np.integer):
        data_iw = data[:,:,neuron_nb]
    else:
        raise ValueError('neuron_nb must be int or "all"')
        
        
    data_delta_iw = data_iw[1:,:] - data_iw[:-1,:]       # delta
    ratio = data_delta_iw / data_iw[:-1,:]               # delta / weight
    ratio_abs = np.abs(ratio)
    
    if axis is None:
        fig, axis = plt.subplots(1,1, figsize=figsize)
    
    if mode == 'raw':
        axis.plot(data_delta_iw, alpha=max(1/nw, 0.002), color='red');
    else:
        if mode == 'full':
            axis.plot(ratio_abs, alpha=max(1/nw, 0.002), color='pink');
            axis.set_title(title + ' raw # ' + str(neuron_nb))

        if mode == 'median':
            axis.plot(np.min(ratio_abs, axis=-1), alpha=.05, color=color);
            axis.plot(np.max(ratio_abs, axis=-1), alpha=.05, color=color);
            axis.plot(np.percentile(ratio_abs, 10, axis=-1), alpha=.3, color=color);
            axis.plot(np.percentile(ratio_abs, 90, axis=-1), alpha=.3, color=color);
            axis.plot(np.median(ratio_abs, axis=-1), alpha=1, color=color);

        if mode == 'norms':
            data_norm_i = np.linalg.norm(data_iw, axis=-1)
            data_delta_norm_i = np.linalg.norm(data_delta_iw, axis=-1)
            axis.plot(data_delta_norm_i/data_norm_i[:-1], alpha=1, color=color)

        if mode == 'dl4j':
            tmp_ = np.mean(np.abs(data_delta_iw), axis=-1) / np.mean(np.abs(data_iw[:-1,:]), axis=1)
            axis.plot(tmp_, color=color)

        if mode == 'mean+std':
            mean_ = np.mean(ratio_abs, axis=-1)
            axis.plot(mean_, alpha=1, color='orange');
            std_ = np.std(ratio_abs, axis=-1)
            axis.plot(mean_ + std_, alpha=.3, color='orange');

        axis.plot([0,len(ratio_abs)],[.1, .1], ls='--', c='black')
        axis.plot([0,len(ratio_abs)],[.01, .01], ls='-', c='black')
        axis.plot([0,len(ratio_abs)],[.001, .001], ls='--', c='black')
        axis.set_yscale('log')
        
        if title is not None:
            axis.set_title(title + ' ' + mode + ' # ' + str(neuron_nb))
            
            
            
def plot_3d_histogram(data, es, neuron_nb, funct=None, 
                      skip_first=False, lines_01=True, title=None, color=(1,0,0,1), ax=None, figsize=None):
    assert data.ndim == 3
    
    if funct is None:
        funct = lambda x: x
        
    if isinstance(funct, str):
        if funct == 'sigmoid': funct = lambda x: 1 / (1 + np.exp(-x))
        elif funct == 'tanh': funct = lambda x: np.tanh(x)
        elif funct == 'softsign': funct = lambda x: x / (1+np.abs(x))
        elif funct == 'relu': funct = lambda x: np.maximum(0, x)
        elif funct == 'lrelu': funct = lambda x: np.where(x > 0, x, x * 0.01)
        else: raise ValueError('Unknown function string')
    
    ni, na, nn = data.shape  # iter, activations (batch size), neurons
    
    if neuron_nb == 'all':
        #data_ia = data.reshape([ni, -1])
        data_ia = data.reshape([-1, es*nn])
    elif np.issubdtype(type(neuron_nb), np.integer):
        #data_ia = data[:,:,neuron_nb]
        data_ia = data[:,:,neuron_nb].reshape(-1, es)
    else:
        raise ValueError('neuron_nb must be int or "all"')
        
    
    
    def interpolate_colors(cstart, cend, n):
        cstart, cend = np.array(cstart), np.array(cend)
        assert cstart.shape == (4,)
        assert cend.shape == (4,)
        if n == 1:  return cend    # if one step, then return end color

        cols = []
        for i in range(n):
            step = i/(n-1)
            cols.append( (1-step)*cstart + step*cend)
        return np.array(cols)
    
    color = np.array(color)
    color_start = np.array(color/4, dtype=float)  # transparent black
    color_end = np.array(color)
    colors = interpolate_colors(color_start, color_end, len(data_ia))
    

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    
    
    ax.view_init(30, -85)
    
    for epoch in range(len(data_ia)):
        
        hist_0 = np.count_nonzero(data_ia[epoch,:]>0) / len(data_ia[epoch,:])
        
        data = funct(data_ia[epoch,:])
        
        if skip_first:
            data = data[data>0]
        
        hist, bins = np.histogram(data, bins=100)
        bins = (bins[:-1] + bins[1:])/2
        if np.sum(hist) != 0:
            hist = hist / np.sum(hist)
        
        ax.plot(xs=bins, ys=hist,
                zs=-epoch,
                zdir='y', 
                color=colors[epoch])
        nb_epochs = len(data_ia)
        if epoch == 0 and lines_01:
            ax.plot(xs=[0,0], ys=[0,0], zs=[-nb_epochs,0], zdir='y', color='k')
            ax.plot(xs=[1,1], ys=[0,0], zs=[-nb_epochs,0], zdir='y', color='k', ls='--')
        if epoch == len(data_ia)-1:
            ax.plot(xs=[bins[0],bins[-1]], ys=[0,0], zs=-nb_epochs, zdir='y', color='k')
    
    if skip_first:
        ax.set_xlabel('value ('+str(round(hist_0*100, 2)) + '%)'); ax.set_ylabel('epoch'); ax.set_zlabel('n')
    else:
        ax.set_xlabel('value'); ax.set_ylabel('epoch'); ax.set_zlabel('n')
        
    if title is not None:
        ax.set_title(title)