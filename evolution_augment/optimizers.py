import dataclasses

import torch
from transformers import optimization

def _add_weight_decay(params, weight_decay=1e-5):
    '''
    reference: https://github.com/huggingface/pytorch-image-models/blob/1bb3989b61c083f89a362f6d1f2c60af35c36203/timm/optim/optim_factory.py#L45
    '''
    decay = []
    no_decay = []
    for param in params:
        if not param.requires_grad:
            continue
        if len(param.shape) == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_optimizer(params, config):
    params = _add_weight_decay(params, weight_decay=config.weight_decay)
    if config.__class__.__name__ == 'SGD':
        config = dataclasses.asdict(config)
        print('Building SGD optimizer with config:\n{}'.format(config))
        return torch.optim.SGD(params=params, **config)
    else:
        raise ValueError('Unsupported optimizer requested')


def get_lr_scheduler(optimizer, config):
    config = dataclasses.asdict(config)
    print('Building LR scheduler with config:\n{}'.format(config))
    return optimization.get_scheduler(optimizer=optimizer, **config)
