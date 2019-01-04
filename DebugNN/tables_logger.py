import numpy as np
import tables
import pdb

def create_log(filename, all_layers_dict, batch_size_to_save):
    with tables.open_file(filename, mode='w') as f:
        for layer_name, layer_dict in all_layers_dict.items():
            group = f.create_group('/', layer_name)
            for tensor_name, tensor in layer_dict.items():
                tensor_shape = tensor.get_shape().as_list()
                if len(tensor_shape) > 0 and tensor_shape[0] is None:
                    tensor_shape[0] = batch_size_to_save
                f.create_earray(group, tensor_name, atom=tables.Float32Atom(), shape=[0, *tensor_shape])
                # print(layer_name, tensor_name, tensor_shape)

                
def print_raw(filename):
    with tables.open_file(filename, mode='r') as f:
        print(f)
    
    
def print_log(filename):
    with tables.open_file(filename, mode='a') as f:
        for group in f.root:
            print(group._v_name)
            for tensor_earray in group:
                print('  {0:10} {1:20} {2:30}'.format(group._v_name, tensor_earray._v_name, str(tensor_earray.shape)),
                      '{0:12.2f}'.format(tensor_earray[:].nbytes/1e6), 'MB')
    
    
def append_log(filename, all_layers_dict):
    with tables.open_file(filename, mode='a') as f:
        for layer_name, layer_dict in all_layers_dict.items():
            group = f.root[layer_name]
            for tensor_name, tensor_data in layer_dict.items():
                group[tensor_name].append( np.expand_dims(tensor_data, axis=0) )
                

def extract_layer_and_flatten(group):
    result = {
        'W_raw': np.array(group.W),
        'b_raw': np.array(group.b),
        'dW_raw': np.array(group.dW),
        'db_raw': np.array(group.db),
        'z_raw': np.array(group.z),
    }
    
    ni, nn = result['W_raw'].shape[0], result['W_raw'].shape[-1]
    result['W'] = result['W_raw'].reshape([ni, -1, nn])
    result['dW'] = result['dW_raw'].reshape([ni, -1, nn])
    result['b'] = result['b_raw']
    result['db'] = result['db_raw']
    
    ni, nn = result['z_raw'].shape[0], result['z_raw'].shape[-1]
    result['z'] = result['z_raw'].reshape([ni, -1, nn])
    
    return result
