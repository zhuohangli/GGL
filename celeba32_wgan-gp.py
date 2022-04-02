import numpy as np
import argparse
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import copy
import lpips
import inversefed
import os
import pandas as pd
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, convert_to_images)
from constants import *
import defense

import nevergrad as ng
from tqdm import tqdm

from turbo import Turbo1



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
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use.')
parser.add_argument('--out_dir', type=str, default='out/celeba-32/gan')
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)



class Generator(nn.Module):
    def __init__(self, DIM=128):
        super(Generator, self).__init__()
        self.DIM = DIM

        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, z):
        DIM = self.DIM
        output = self.preprocess(z)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


def adam_reconstruction(model, generator, input_gradient, dm, ds, num_images=1, args=args):
    config = dict(signed=True,
                  boxed=False,
                  cost_fn='l2',
                  indices='def',
                  weights='equal',
                  lr=1e-1,
                  optim='adam',
                  restarts=args.n_trials,
                  max_iterations=args.budget, # 2500
                  total_variation=0,
                  init='randn',
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss',
                  optim_noise=0,
                  KL=1e-1, # use 1e-4 for sim loss and 1e-1 for l2 loss
                  l2_penalty=0,
                  EOT_DP=0.0,
                  EOT_C=0,
                  perturb_rep=0)

    rec_machine = inversefed.GeneratorBasedGradientReconstructor(model, generator, (dm, ds), config, num_images=num_images)
    output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=(128,))

    return output, stats['opt']


class BOReconstructor():
    """
    BO Reconstruction for WGAN-GP

    """
    def __init__(self, fl_model, generator, loss_fn, num_classes=2, search_dim=(128,), strategy='BO', budget=1000, use_tanh=False, defense_setting=None):

        self.generator = generator
        self.budget = budget
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.num_samples = 50
        self.defense_setting = defense_setting

        self.grad_steps = 0
        self.grad_lr = 1e-6

        self.fl_setting = {'loss_fn':loss_fn, 'fl_model':fl_model, 'num_classes':num_classes}



    def evaluate_loss(self, z, labels, input_gradient):
        z = torch.Tensor(z).unsqueeze(0).to(input_gradient[0].device)
        return self.ng_loss(z=z, input_gradient=input_gradient, metric='l2',
                        labels=labels, generator=self.generator,
                        use_tanh=self.use_tanh, defense_setting=self.defense_setting, **self.fl_setting
                       ).item()

    def reconstruct(self, input_gradient, use_pbar=True):

        labels = self.infer_label(input_gradient)
        print('Inferred label: {}'.format(labels))
        
        if self.defense_setting is not None:
            if 'clipping' in self.defense_setting:
                total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in input_gradient]), 2)
                self.defense_setting['clipping'] = total_norm.item()
                print('Estimated defense parameter: {}'.format(self.defense_setting['clipping']))
            if 'compression' in self.defense_setting:
                n_zero, n_param = 0, 0
                for i in range(len(input_gradient)):
                    n_zero += torch.sum(input_gradient[i]==0)
                    n_param += torch.numel(input_gradient[i])
                self.defense_setting['compression'] = 100 * (n_zero/n_param).item()
                print('Estimated defense parameter: {}'.format(self.defense_setting['compression']))

        z_lb = -2*np.ones(self.search_dim) # lower bound, you may change -10 to -inf
        z_ub = 2*np.ones(self.search_dim) # upper bound, you may change 10 to inf

        f = lambda z:self.evaluate_loss(z, labels, input_gradient)

        self.optimizer = Turbo1(
                                f=f,  # Handle to objective function
                                lb=z_lb,  # Numpy array specifying lower bounds
                                ub=z_ub,  # Numpy array specifying upper bounds
                                n_init=256,  # Number of initial bounds from an Latin hypercube design
                                max_evals = self.budget,  # Maximum number of evaluations
                                batch_size=10,  # How large batch size TuRBO uses
                                verbose=True,  # Print information from each batch
                                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                                n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                                min_cuda=1024,  # Run on the CPU for small datasets
                                device="cuda", #next(generator.parameters()).device,  # "cpu" or "cuda"
                                dtype="float32",  # float64 or float32
                            )

        self.optimizer.optimize()

        X = self.optimizer.X  # Evaluated points of z
        fX = self.optimizer.fX  # Observed values of ng_loss
        ind_best = np.argmin(fX)
        loss_res, z_res = fX[ind_best], X[ind_best, :]

        loss_res = self.evaluate_loss(z_res, labels, input_gradient)
        z_res = torch.from_numpy(z_res).unsqueeze(0).to(input_gradient[0].device)
        if self.use_tanh:
            z_res = z_res.tanh()

        with torch.no_grad():
            x_res = self.generator(z_res.float())
        img_res = convert_to_images(x_res.cpu())

        return z_res, x_res, img_res, loss_res

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):
        last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
        return labels

    @staticmethod
    def ng_loss(z, # latent variable to be optimized
                loss_fn, # loss function for FL model
                input_gradient,
                labels,
                generator,
                fl_model,
                num_classes=2,
                metric='l2',
                use_tanh=True,
                no_grad=True,
                defense_setting=None # adaptive attack against defense
               ):

        if use_tanh:
            z = z.tanh()

        if no_grad:
            with torch.no_grad():
                x = generator(z)
        else:
            x = generator(z)

        # compute the trial gradient
        fl_model.zero_grad()
        target_loss, _, _ = loss_fn(fl_model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters())
        trial_gradient = [grad.detach() for grad in trial_gradient]

        # adaptive attack against defense
        if defense_setting is not None:
            if 'noise' in defense_setting:
                trial_gradient = defense.additive_noise(trial_gradient, std=defense_setting['noise'])
            if 'clipping' in defense_setting:
                trial_gradient = defense.gradient_clipping(trial_gradient, bound=defense_setting['clipping'])
            if 'compression' in defense_setting:
                trial_gradient = defense.gradient_compression(trial_gradient, percentage=defense_setting['compression'])
            if 'representation' in defense_setting: # for ResNet
                mask = input_gradient[-2][0]!=0
                trial_gradient[-2] = trial_gradient[-2] * mask

        # calculate l2 norm
        dist = 0
        for i in range(len(trial_gradient)):
            if metric == 'l2':
                dist += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()
            elif metric == 'l1':
                dist += ((trial_gradient[i] - input_gradient[i]).abs()).sum()
        dist /= len(trial_gradient)

#         if not use_tanh:
#             KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(), axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))

#             dist += 0.1*KLD

        return dist



class NGReconstructor():
    """
    Reconstruction for WGAN-GP

    """
    def __init__(self, fl_model, generator, loss_fn, num_classes=2, search_dim=(128,), strategy='CMA', budget=500, use_tanh=True, defense_setting=None):

        self.generator = generator
        self.budget = budget
        self.search_dim = search_dim
        self.use_tanh = use_tanh
        self.num_samples = 50
        self.defense_setting = defense_setting

        # parametrization = ng.p.Array(shape=search_dim)
        parametrization = ng.p.Array(init=np.zeros(search_dim))#.set_mutation(sigma=1.0)
        self.ng_optimizer = ng.optimizers.registry[strategy](parametrization=parametrization, budget=budget)

        self.grad_steps = 0
        self.grad_lr = 1e-6

        self.fl_setting = {'loss_fn':loss_fn, 'fl_model':fl_model, 'num_classes':num_classes}



    def evaluate_loss(self, z, labels, input_gradient):
        z = torch.Tensor(z).unsqueeze(0).to(input_gradient[0].device)
        return self.ng_loss(z=z, input_gradient=input_gradient, metric='l2',
                        labels=labels, generator=self.generator,
                        use_tanh=self.use_tanh, defense_setting=self.defense_setting, **self.fl_setting
                       ).item()

    def reconstruct(self, input_gradient, use_pbar=True):

        labels = self.infer_label(input_gradient)
        print('Inferred label: {}'.format(labels))
        
        if self.defense_setting is not None:
            if 'clipping' in self.defense_setting:
                total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in input_gradient]), 2)
                self.defense_setting['clipping'] = total_norm.item()
                print('Estimated defense parameter: {}'.format(self.defense_setting['clipping']))
            if 'compression' in self.defense_setting:
                n_zero, n_param = 0, 0
                for i in range(len(input_gradient)):
                    n_zero += torch.sum(input_gradient[i]==0)
                    n_param += torch.numel(input_gradient[i])
                self.defense_setting['compression'] = 100 * (n_zero/n_param).item()
                print('Estimated defense parameter: {}'.format(self.defense_setting['compression']))
        
        pbar = tqdm(range(self.budget)) if use_pbar else range(self.budget)

        for r in pbar:
            ng_data = [self.ng_optimizer.ask() for _ in range(self.num_samples)]
            loss = [self.evaluate_loss(z=ng_data[i].value, labels=labels, input_gradient=input_gradient) for i in range(self.num_samples)]
            for z, l in zip(ng_data, loss):
                self.ng_optimizer.tell(z, l)

            if use_pbar:
                pbar.set_description("Loss {:.6}".format(np.mean(loss)))
            else:
                print("Round {} - Loss {:.6}".format(r, np.mean(loss)))

        if self.grad_steps > 0:
            print('Gradient-based finetuning.')
            pbar = tqdm(range(self.grad_steps)) if use_pbar else range(self.grad_steps)
            recommendation = self.ng_optimizer.provide_recommendation()
#             z = torch.from_numpy(recommendation.value).unsqueeze(0).float().to(input_gradient[0].device).requires_grad_(True)
            z = torch.tensor(np.expand_dims(recommendation.value, axis=0), dtype=torch.float32, device=input_gradient[0].device, requires_grad=True)
            self.grad_optimizer = torch.optim.Adam([z], lr=self.grad_lr)
            for r in pbar:
                self.grad_optimizer.zero_grad()
                self.generator.zero_grad()
                loss = self.ng_loss(z=z, input_gradient=input_gradient, metric='l2',
                            labels=labels, generator=self.generator,
                            use_tanh=self.use_tanh, no_grad=False, defense_setting=self.defense_setting, **self.fl_setting
                           )
                loss.backward()
                self.grad_optimizer.step()

                if use_pbar:
                    pbar.set_description("Loss {:.6}".format(loss.item()))
                else:
                    print("Round {} - Loss {:.6}".format(r, loss.item()))

            z_res = z.requires_grad_(False)
        else:
            recommendation = self.ng_optimizer.provide_recommendation()
            z_res = torch.from_numpy(recommendation.value).unsqueeze(0).to(input_gradient[0].device)

        if self.use_tanh:
            z_res = z_res.tanh()
        loss_res = self.evaluate_loss(z_res.clone().squeeze().cpu().numpy(), labels, input_gradient)
        with torch.no_grad():
            x_res = self.generator(z_res.float())
        img_res = convert_to_images(x_res.cpu())



        return z_res, x_res, img_res, loss_res

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):
        last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
        return labels

    @staticmethod
    def ng_loss(z, # latent variable to be optimized
                loss_fn, # loss function for FL model
                input_gradient,
                labels,
                generator,
                fl_model,
                num_classes=2,
                metric='l2',
                use_tanh=True,
                no_grad=True,
                defense_setting=None # adaptive attack against defense
               ):

        if use_tanh:
            z = z.tanh()

        if no_grad:
            with torch.no_grad():
                x = generator(z)
        else:
            x = generator(z)

        # compute the trial gradient
        fl_model.zero_grad()
        target_loss, _, _ = loss_fn(fl_model(x), labels)
        trial_gradient = torch.autograd.grad(target_loss, fl_model.parameters())
        trial_gradient = [grad.detach() for grad in trial_gradient]

        # adaptive attack against defense
        if defense_setting is not None:
            if 'noise' in defense_setting:
                trial_gradient = defense.additive_noise(trial_gradient, std=defense_setting['noise'])
            if 'clipping' in defense_setting:
                trial_gradient = defense.gradient_clipping(trial_gradient, bound=defense_setting['clipping'])
            if 'compression' in defense_setting:
                trial_gradient = defense.gradient_compression(trial_gradient, percentage=defense_setting['compression'])
            if 'representation' in defense_setting: # for ResNet
                mask = input_gradient[-2][0]!=0
                trial_gradient[-2] = trial_gradient[-2] * mask

        # calculate l2 norm
        dist = 0
        for i in range(len(trial_gradient)):
            if metric == 'l2':
                dist += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum()
            elif metric == 'l1':
                dist += ((trial_gradient[i] - input_gradient[i]).abs()).sum()
        dist /= len(trial_gradient)

        if not use_tanh:
            KLD = -0.5 * torch.sum(1 + torch.log(torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(z.squeeze(), axis=-1).pow(2) - torch.std(z.squeeze(), unbiased=False, axis=-1).pow(2))

            dist += 0.1*KLD

        return dist




def run_exp(args=args):

    # ----------- initialization --------------

    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    defs.augmentation = False
    device = setup['device']

    dm = torch.as_tensor([0.5, 0.5, 0.5], **setup)[:, None, None]
    ds = torch.as_tensor([0.5, 0.5, 0.5], **setup)[:, None, None]

    loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CelebA-32', defs, data_path=celeba_path, normalize=True)

    torch.manual_seed(args.seed)
#     model = torchvision.models.resnet18(pretrained=args.trained_model)
    model, _ = inversefed.construct_model('ResNet18', num_classes=2, num_channels=3, seed=args.seed)
    model.to(**setup)

    if args.trained_model:
        epochs = 120
        file = f'{arch}_{epochs}.pth'
        try:
            model.load_state_dict(torch.load(f'models/{file}'))
        except FileNotFoundError:
            inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)
            torch.save(model.state_dict(), f'models/{file}')

    model.eval()

    generator = Generator()
    save_dir = celeba_gan_path
    checkpoint = torch.load(save_dir)
    generator.load_state_dict(checkpoint['state_dict'])
    generator.eval()

    generator.to(device)

    # ----------- compute input gradient --------------

    img, label = validloader.dataset[args.idx]
    labels = torch.as_tensor((label,), device=device)
    ground_truth = img.to(**setup).unsqueeze(0)
    print('Using the #{} image from the validation set.'.format(args.idx))
    print('Original label: {}'.format(labels))

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

    if args.ng_method == 'adam':
        z_res, loss_res = adam_reconstruction(model, generator, input_gradient, dm, ds, num_images=1, args=args)
        x_res = generator(z_res.float())
        img_res = convert_to_images(x_res.cpu())
        # create a dummy Reconstructor
        ng_rec = BOReconstructor(fl_model=model, generator=generator, loss_fn=loss_fn,
                                                     num_classes=2, search_dim=(128,), budget=args.budget, use_tanh=False, defense_setting=defense_setting)
    else:
        for t in range(args.n_trials):
            print('Processing trial {}/{}.'.format(t+1, args.n_trials))
            if args.ng_method == 'BO':
                ng_rec = BOReconstructor(fl_model=model, generator=generator, loss_fn=loss_fn,
                                                     num_classes=2, search_dim=(128,), budget=args.budget, use_tanh=False, defense_setting=defense_setting)
                z_res, x_res, img_res, loss_res = ng_rec.reconstruct(input_gradient)
            else:
                ng_rec = NGReconstructor(fl_model=model, generator=generator, loss_fn=loss_fn,
                                         num_classes=2, search_dim=(128,), strategy=args.ng_method, budget=args.budget, use_tanh=False, defense_setting=defense_setting)
                z_res, x_res, img_res, loss_res = ng_rec.reconstruct(input_gradient)
            res_trials[t] = {'z':z_res, 'x':x_res, 'img':img_res}
            loss_trials[t] = loss_res

        best_t = np.argmin(loss_trials)
        z_res, x_res, img_res = res_trials[best_t]['z'], res_trials[best_t]['x'], res_trials[best_t]['img']

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
    columns = ['idx', 'labels', 'ng_method', 'budget', 'grad_steps', 'grad_lr', 'defense', 'd_param', 'trained_model', 'seed', 'n_trials', 'adaptive', 'loss', 'MSE', 'LPIPS(VGG)', 'LPIPS(ALEX)', 'PSNR', 'RMSE', 'FMSE', 'TV-ori', 'TV-rec']
    data = pd.DataFrame(columns = columns)

    data = data.append({
    'idx': args.idx,
    'labels': label,
    'ng_method': args.ng_method,
    'budget': args.budget,
    'grad_steps': ng_rec.grad_steps,
    'grad_lr': ng_rec.grad_lr,
    'defense': args.defense,
    'd_param': d_param,
    'trained_model': args.trained_model,
    'seed': args.seed,
    'n_trials': args.n_trials,
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
