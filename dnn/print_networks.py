import torch

def print_networks(name, net, verbose):
    def num_params_str(num_params):
        if num_params < 1_000_000:
            return f'{num_params/1e3:.3f} K'
        return f'{num_params/1e6:.3f} M'

    if isinstance(net, torch.Tensor):
        num_params = net.numel()
        return f'[Tensor {name}] Total number of parameters : {num_params_str(num_params)}'
    else:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()

        ret_msg = ''
        if verbose:
            ret_msg = f'{net}\n'

        ret_msg += f'[Network {name}] Total number of parameters : {num_params_str(num_params)}'
        return ret_msg
