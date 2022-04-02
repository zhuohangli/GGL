import numpy as np
import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import lpips
import inversefed
import os
import pandas as pd
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, convert_to_images)
from reconstructor import NGReconstructor, BOReconstructor, AdamReconstructor
import defense
from constants import *





parser = argparse.ArgumentParser()
parser.add_argument('--ng_method', type=str, default='CMA', help='Type of optimizer to use, can be CMA, BO, adam, or other optimizers supported in nevergrad.')
parser.add_argument('--idx', type=int, default=4000)
parser.add_argument('--defense', type=str, default=None, choices=[None, 'compression', 'noise', 'clipping', 'representation'])
parser.add_argument('--d_param', type=float, default=None, help='Parameter setting for the defense, i.e., std for noise, bound for clipping, and pruning rate for others.')
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--budget', type=int, default=500, help='Budget for the attack.')
parser.add_argument('--n_trials', type=int, default=1)
parser.add_argument('--model', type=str, default='ResNet18')
parser.add_argument('--trained_model', action='store_true')
parser.add_argument('--use_weight', action='store_true')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use.')
parser.add_argument('--out_dir', type=str, default='out/imagenet/gan')
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)



def run_exp(args=args):
    
    # ----------- initialization --------------
    
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    device = setup['device']
    
    dm = torch.as_tensor([0.5, 0.5, 0.5], **setup)[:, None, None]
    ds = torch.as_tensor([0.5, 0.5, 0.5], **setup)[:, None, None]

    loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('ImageNet', defs, 
                                                                          data_path=imagenet_path)
    torch.manual_seed(args.seed)
    model = torchvision.models.resnet18(pretrained=args.trained_model)
#     model, _ = inversefed.construct_model('ResNet18', num_classes=1000, num_channels=3, seed=123)
    model.to(**setup)
    model.eval()
    
    generator= BigGAN.from_pretrained('biggan-deep-256')
    generator.to(device)
    
    # ----------- compute input gradient --------------
    
    img, label = validloader.dataset[args.idx]
    labels = torch.as_tensor((label,), device=device)
    ground_truth = img.to(**setup).unsqueeze(0)
    #     plot(ground_truth)
    print('Using the #{} image from the validation set.'.format(args.idx))
    print('Original label: {} ({})'.format(labels, [trainloader.dataset.classes[l] for l in labels]))
    
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]
    full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
    print('Input gradient norm: {}'.format(full_norm))
    
    # ----------- apply defense --------------

    defense_method = args.defense
    if defense_method is None:
        print('No defense applied.')
        d_param = args.d_param
    else:
        if defense_method == 'noise':
            d_param = 0.01 if args.d_param is None else args.d_param
            input_gradient = defense.additive_noise(input_gradient, std=d_param)
        elif defense_method == 'clipping':
            d_param = 4 if args.d_param is None else args.d_param
            input_gradient = defense.gradient_clipping(input_gradient, bound=d_param)
        elif defense_method == 'compression':
            d_param = 20 if args.d_param is None else args.d_param
            input_gradient = defense.gradient_compression(input_gradient, percentage=d_param)
        elif defense_method == 'representation':
            d_param = 10 if args.d_param is None else args.d_param
            input_gradient = defense.perturb_representation(input_gradient, model, ground_truth, pruning_rate=d_param)
        else:
            raise NotImplementedError("Invalid defense method!")
        print('Defense applied: {} w/ {}.'.format(defense_method, d_param))
    
    # ----------- GAN-based reconstruction --------------
    
    print()
    print('-'*20)
    print('Reconstructing original image using GAN-based method.')
    
    res_trials = [None]*args.n_trials
    loss_trials = [None]*args.n_trials
    
    if args.adaptive:
        print('Using adaptive attack.')
        defense_setting = dict()
        defense_setting[defense_method] = d_param
    else:
        defense_setting = None
    
    for t in range(args.n_trials):
        print('Processing trial {}/{}.'.format(t+1, args.n_trials))
        if args.ng_method == 'BO':
            ng_rec = BOReconstructor(fl_model=model, generator=generator, loss_fn=loss_fn,
                                                 num_classes=1000, search_dim=(128,), strategy=args.ng_method, budget=args.budget, use_tanh=False, use_weight=args.use_weight, defense_setting=defense_setting)
            z_res, x_res, img_res, loss_res = ng_rec.reconstruct(input_gradient)
        elif args.ng_method == 'adam':
            ng_rec = AdamReconstructor(fl_model=model, generator=generator, loss_fn=loss_fn,
                                                 num_classes=1000, search_dim=(128,), strategy=args.ng_method, budget=args.budget, use_tanh=False, use_weight=args.use_weight, defense_setting=defense_setting)
            z_res, x_res, img_res, loss_res = ng_rec.reconstruct(input_gradient)
        else:
            ng_rec = NGReconstructor(fl_model=model, generator=generator, loss_fn=loss_fn,
                                     num_classes=1000, search_dim=(128,), strategy=args.ng_method, budget=args.budget, use_tanh=True, use_weight=args.use_weight, defense_setting=defense_setting)
            z_res, x_res, img_res, loss_res = ng_rec.reconstruct(input_gradient)
        res_trials[t] = {'z':z_res, 'x':x_res, 'img':img_res}
        loss_trials[t] = loss_res
    
    best_t = np.argmin(loss_trials)
    z_res, x_res, img_res = res_trials[best_t]['z'], res_trials[best_t]['x'], res_trials[best_t]['img']
    loss_res = loss_trials[best_t]
    
    print('GAN-based final loss: {}'.format(loss_res))
    print('z mean: {}, std:{}'.format(z_res.mean(), z_res.std()))
    
    # ----------- compute scores --------------
    
    lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(device)
    lpips_loss_a = lpips.LPIPS(net='alex', spatial=False).to(device)
    
    with torch.no_grad():
        lpips_score = lpips_loss(x_res, ground_truth).squeeze().item()
        lpips_score_a = lpips_loss_a(x_res, ground_truth).squeeze().item()
        feat_mse = (model(x_res.detach()) - model(ground_truth)).pow(2).mean().item()
    
    mse = (x_res - ground_truth).pow(2).mean().item()
    psnr_score = inversefed.metrics.psnr(img_batch=x_res, ref_batch=ground_truth)
    tv_o = inversefed.metrics.total_variation(ground_truth).item()
    tv_r = inversefed.metrics.total_variation(x_res).item()
    
    # calculate the MSE of representations
    catted_inputs = torch.cat((x_res, ground_truth), dim=0)
    separator = x_res.shape[0]
    rep_data = {}
    def get_RMSE(model, input, output):
        layer_inputs = input[0].detach()
        rep_data['rmse'] = (layer_inputs[:separator] - layer_inputs[separator:]).pow(2).mean().item()
    handle = model.fc.register_forward_hook(get_RMSE) # for ResNet-18
    with torch.no_grad():
        out = model(catted_inputs)
    handle.remove()
    rep_mse = rep_data['rmse']
    
    print('LPIPS score (VGG): {:.3f}, LPIPS score (ALEX): {:.3f}, MSE: {:.5f}, PSNR: {:.3f}, FMSE: {:.5f}, RMSE: {:.5f}, TV of original: {:.3f}, TV of reconstructed: {:.3f}'.format(lpips_score, lpips_score_a, mse, psnr_score, feat_mse, rep_mse, tv_o, tv_r))
    
    
    
    # ----------- save output files --------------
    
    if not os.path.exists(os.path.join(args.out_dir, args.exp_name)):
        os.makedirs(os.path.join(args.out_dir, args.exp_name))
        save_dir = os.path.join(args.out_dir, args.exp_name)
    else:
        save_dir = os.path.join(args.out_dir, args.exp_name+'_1')
        while os.path.exists(save_dir):
            save_dir += '_1'
        os.makedirs(save_dir)
        
    original_img = ground_truth.mul_(ds).add_(dm).clamp_(0, 1).mul_(255).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    res_img = x_res.mul_(ds).add_(dm).clamp_(0, 1).mul_(255).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    plt.imsave(os.path.join(save_dir, 'original.png'), original_img)
    plt.imsave(os.path.join(save_dir, 'GAN_out.png'), res_img)
    np.save(os.path.join(save_dir, 'z.npy'), z_res.clone().cpu().numpy())
    
    # log file
    columns = ['idx', 'labels', 'ng_method', 'defense', 'd_param', 'trained_model', 'seed', 'n_trials', 'budget', 'adaptive', 'loss', 'MSE', 'LPIPS(VGG)', 'LPIPS(ALEX)', 'PSNR', 'RMSE', 'FMSE', 'TV-ori', 'TV-rec']  
    data = pd.DataFrame(columns = columns)
    
    data = data.append({
    'idx': args.idx,
    'labels': label,
    'ng_method': args.ng_method,
    'defense': args.defense,
    'd_param': d_param,
    'trained_model': args.trained_model,
    'seed': args.seed,
    'n_trials': args.n_trials,
    'budget': args.budget,
    'adaptive': args.adaptive,
    'loss': loss_res,
    'MSE': mse,
    'LPIPS(VGG)': lpips_score,
    'LPIPS(ALEX)': lpips_score_a,
    'PSNR': psnr_score,
    'FMSE': feat_mse,
    'RMSE': rep_mse,
    'TV-ori': tv_o,
    'TV-rec': tv_r,
    },ignore_index=True)
    data.to_csv(os.path.join(save_dir, 'log.csv'))
    
    print('Files saved at {}'.format(save_dir))
    
    


if __name__ == '__main__':
    run_exp()