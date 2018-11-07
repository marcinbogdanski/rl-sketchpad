

import numpy as np

import tiles3           # by Richard Sutton, http://incompleteideas.net/tiles/tiles3.html



class KerasFunctApprox():

    def __init__(self, model, st_low, st_high, nb_actions):
        """Q-function approximator using Keras model

        Args:
            model: Keras compiled model
        """
        st_low = np.array(st_low);    st_high = np.array(st_high)
        self._model = model
    
        first_layer = self._model.layers[0]
        nn_input_shape = first_layer.input_shape[1:]
        if st_low.shape != nn_input_shape:
            raise ValueError('Input shape does not match state_space shape')

        last_layer = self._model.layers[-1]
        nn_output_shape = last_layer.output_shape[1:]
        if (nb_actions,) != nn_output_shape:
            raise ValueError('Output shape does not match action_space shape')

        # normalise inputs
        self._offsets = st_low + (st_high - st_low) / 2
        self._scales = 1 / ((st_high - st_low) / 2)

    def eval(self, states):
        assert isinstance(states, np.ndarray)
        assert states.ndim == 2

        inputs = (states - self._offsets) * self._scales

        return self._model.predict(inputs, batch_size=len(inputs))

    def train(self, states, actions, targets):
        
        assert isinstance(states, np.ndarray)
        assert isinstance(actions, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert states.ndim == 2
        assert actions.ndim == 1
        assert targets.ndim == 1
        assert len(states) == len(actions) == len(targets)

        inputs = (states - self._offsets) * self._scales
        all_targets = self._model.predict(inputs, batch_size=len(inputs))
        all_targets[np.arange(len(all_targets)), actions] = targets
        self._model.fit(inputs, all_targets, batch_size=len(inputs), epochs=1, verbose=False)




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