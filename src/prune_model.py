import torch
from torch.nn.utils import prune
def get_total_parameters_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pruned_parameters_count(pruned_model):
    params = 0
    for param in pruned_model.parameters():
        if param is not None:
            params += torch.nonzero(param).size(0)
    return params

def prune_model(model,reduce_size = 0.25):
    total_params_count = get_pruned_parameters_count(model)
    pruning_percentage = reduce_size  # Corrected variable name
    parameters_to_prune = (
        (model.enc_blocks[0].target.vmapped.conv, 'weight'),
        (model.enc_blocks[0].support.vmapped.conv, 'weight'),
        (model.enc_blocks[1].cross.cross_conv, 'weight'),
        (model.enc_blocks[1].target.vmapped.conv, 'weight'),
        (model.enc_blocks[1].support.vmapped.conv, 'weight'),
        (model.enc_blocks[2].cross.cross_conv, 'weight'),
        (model.enc_blocks[2].target.vmapped.conv, 'weight'),
        (model.enc_blocks[2].support.vmapped.conv, 'weight'),
        (model.enc_blocks[3].cross.cross_conv, 'weight'),
        (model.enc_blocks[3].target.vmapped.conv, 'weight'),
        (model.enc_blocks[3].support.vmapped.conv, 'weight'),
        (model.dec_blocks[0].cross.cross_conv, 'weight'),
        (model.dec_blocks[0].target.vmapped.conv, 'weight'),
        (model.dec_blocks[0].support.vmapped.conv, 'weight'),
        (model.dec_blocks[1].cross.cross_conv, 'weight'),
        (model.dec_blocks[1].target.vmapped.conv, 'weight'),
        (model.dec_blocks[1].support.vmapped.conv, 'weight'),
        (model.dec_blocks[2].cross.cross_conv, 'weight'),
        (model.dec_blocks[2].target.vmapped.conv, 'weight'),
        (model.dec_blocks[2].support.vmapped.conv, 'weight')
    )

    prune.global_unstructured(parameters_to_prune,pruning_method=prune.L1Unstructured,amount=pruning_percentage)

    prune.remove(model.enc_blocks[0].target.vmapped.conv, 'weight'),
    prune.remove(model.enc_blocks[0].support.vmapped.conv, 'weight'),
    prune.remove(model.enc_blocks[1].cross.cross_conv, 'weight'),
    prune.remove(model.enc_blocks[1].target.vmapped.conv, 'weight'),
    prune.remove(model.enc_blocks[1].support.vmapped.conv, 'weight'),
    prune.remove(model.enc_blocks[2].cross.cross_conv, 'weight'),
    prune.remove(model.enc_blocks[2].target.vmapped.conv, 'weight'),
    prune.remove(model.enc_blocks[2].support.vmapped.conv, 'weight'),
    prune.remove(model.enc_blocks[3].cross.cross_conv, 'weight'),
    prune.remove(model.enc_blocks[3].target.vmapped.conv, 'weight'),
    prune.remove(model.enc_blocks[3].support.vmapped.conv, 'weight'),
    prune.remove(model.dec_blocks[0].cross.cross_conv, 'weight'),
    prune.remove(model.dec_blocks[0].target.vmapped.conv, 'weight'),
    prune.remove(model.dec_blocks[0].support.vmapped.conv, 'weight'),
    prune.remove(model.dec_blocks[1].cross.cross_conv, 'weight'),
    prune.remove(model.dec_blocks[1].target.vmapped.conv, 'weight'),
    prune.remove(model.dec_blocks[1].support.vmapped.conv, 'weight'),
    prune.remove(model.dec_blocks[2].cross.cross_conv, 'weight'),
    prune.remove(model.dec_blocks[2].target.vmapped.conv, 'weight'),
    prune.remove(model.dec_blocks[2].support.vmapped.conv, 'weight')
    pruned_model_param_count = get_pruned_parameters_count(model)
    print('Original Model paramete count:', total_params_count)
    print('Pruned Model parameter count:', pruned_model_param_count)
    print(f'Compressed Percentage: {(100 - (pruned_model_param_count / total_params_count) * 100)}%')
    return model