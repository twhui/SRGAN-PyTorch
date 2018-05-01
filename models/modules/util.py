import os
import torch
import torch.nn as nn

# helper printing function
def get_network_description(network):
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))
    return s, n

# helper saving function
def save_network(save_dir, network, network_label, iter_label, gpu_ids):
    save_filename = '%s_%s.pth' % (iter_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.module.cpu().state_dict(), save_path) # network is wrapped by nn.DataParallel
    network.module.cuda(gpu_ids[0])

# helper loading function
def load_network(load_path, network):
    network.module.load_state_dict(torch.load(load_path),False)
