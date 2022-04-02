"""
Defense methods.

Including:
- Additive noise
- Gradient clipping
- Gradient compression
- Representation perturbation
"""
import numpy as np
import torch



def additive_noise(input_gradient, std=0.1):
    """
    Additive noise mechanism for differential privacy
    """
    gradient = [grad + torch.normal(torch.zeros_like(grad), std*torch.ones_like(grad)) for grad in input_gradient]
    return gradient


def gradient_clipping(input_gradient, bound=4):
    """
    Gradient clipping (clip by norm)
    """
    max_norm = float(bound)
    norm_type = 2.0 # np.inf
    device = input_gradient[0].device
    
    if norm_type == np.inf:
        norms = [g.abs().max().to(device) for g in input_gradient]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for g in input_gradient]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    gradient = [g.mul_(clip_coef_clamped.to(device)) for g in input_gradient]
    return gradient


def gradient_compression(input_gradient, percentage=10):
    """
    Prune by percentage
    """
    device = input_gradient[0].device
    gradient = [None]*len(input_gradient)
    for i in range(len(input_gradient)):
        grad_tensor = input_gradient[i].clone().cpu().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, percentage)
        grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
        gradient[i] = torch.Tensor(grad_tensor).to(device)
    return gradient


def perturb_representation(input_gradient, model, ground_truth, pruning_rate=10):
    """
    Defense proposed in the Soteria paper.
    param:
        - input_gradient: the input_gradient
        - model: the ResNet-18 model
        - ground_truth: the benign image (for learning perturbed representation)
        - pruning_rate: the prune percentage
    Note: This implementation only works for ResNet-18
    """
    device = input_gradient[0].device
    
    gt_data = ground_truth.clone()
    gt_data.requires_grad=True

    # register forward hook to get intermediate layer output
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0]
        return hook

    # for ResNet-18
    handle = model.fc.register_forward_hook(get_activation('flatten'))
    out = model(gt_data)
    
    feature_graph = activation['flatten']

    
    deviation_target = torch.zeros_like(feature_graph)
    deviation_x_norm = torch.zeros_like(feature_graph)
    for f in range(deviation_x_norm.size(1)):
        deviation_target[:,f] = 1
        feature_graph.backward(deviation_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/((feature_graph.data[:,f]) + 1e-10)
        model.zero_grad()
        gt_data.grad.data.zero_()
        deviation_target[:,f] = 0
        
    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_x_norm_sum = deviation_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    mask = np.where(abs(deviation_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
    
    print('Soteria mask: ', sum(mask))

    gradient = [grad for grad in input_gradient]
    # apply mask
    gradient[-2] = gradient[-2] * torch.Tensor(mask).to(device)
    
    handle.remove()
    
    return gradient