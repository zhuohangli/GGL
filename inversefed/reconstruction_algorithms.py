"""Mechanisms for image reconstruction from parameter gradients."""

import torch
from collections import defaultdict, OrderedDict
from inversefed.nn import MetaMonkey
from .metrics import total_variation as TV
from .metrics import InceptionScore
from .medianfilt import MedianPool2d
from copy import deepcopy

import time

DEFAULT_CONFIG = dict(signed=False,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss',
                      l2_reg=0, # the l2 norm as standard image prior
                      l2_norm=1e-2, # the l2 norm used to bound distance between the trial image and the initial image
                      optim_noise=0,
                      KL=0,
                      l2_penalty=0,
                      EOT_DP=0,
                      EOT_C=0,
                      perturb_rep=0) # which layer to apply the representation perturbation defense. 0 means no defense.

def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.iDLG = True

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None, x_init=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()


        stats = defaultdict(list)
        x = self._init_images(img_shape)
        if x_init is not None:
            x.data = x_init.clone().data

        scores = torch.zeros(self.config['restarts'])

        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
                self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            for trial in range(self.config['restarts']):
                if x_init is not None:
                    x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun, x_init=x_init.clone().detach().requires_grad_(True))
                else:
                    x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun, x_init=None)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False, x_init=None):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels])
            else:
                raise ValueError()
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial])
            else:
                raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels, x_init)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    if self.config['optim_noise'] > 0:
                        x_trial.data = x_trial.data + self.config['optim_noise'] * scheduler.get_last_lr()[0] * torch.randn_like(x_trial)
                    # Project into image space
                    if self.config['boxed']:
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label, x_init=None):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

            if self.config['EOT_C'] > 0:
                gradient = [torch.clamp(grad, -self.config['EOT_C'], self.config['EOT_C']) for grad in gradient]
            if self.config['EOT_DP'] > 0:
                gradient =  [grad + torch.normal(torch.zeros_like(grad), self.config['EOT_DP']*torch.ones_like(grad)) for grad in gradient]
            if self.config['perturb_rep'] != 0:
                mask = input_gradient[self.config['perturb_rep']][0]!=0
                gradient = [grad for grad in gradient]
                gradient[self.config['perturb_rep']] = gradient[self.config['perturb_rep']] * mask

            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            if self.config['l2_reg'] > 0:
                rec_loss += self.config['l2_reg'] * (x_trial).pow(2).sum()
            if x_init is not None:
                rec_loss += self.config['l2_norm'] * ((x_trial - x_init).pow(2)).sum()
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(x_optimal).softmax(dim=1)
        loss = self.loss_fn(self.model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats



class GeneratorBasedGradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, generator, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.generator = generator
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.iDLG = True

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None, x_init=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()
            self.generator.eval()


        stats = defaultdict(list)
        x = self._init_images(img_shape)
        if x_init is not None:
            x.data = x_init.clone().data

        scores = torch.zeros(self.config['restarts'])


        # Infer Label information
        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # sum works better than column-min under DP setting
                last_weight_min = torch.argsort(torch.sum(input_data[-2], dim=-1), dim=-1)[:self.num_images]
                labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
                self.reconstruct_label = False
            print("Inferred Labels: {}".format(labels))
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            for trial in range(self.config['restarts']):
                x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        elif self.config['scoring_choice'] == 'last':
            x_optimal = x[-1]
            stats['opt'] = scores[-1].item()
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(self.generator(x_trial))
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels])
            else:
                raise ValueError()
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial])
            else:
                raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    if self.config['optim_noise'] > 0:
                        x_trial.data = x_trial.data + self.config['optim_noise'] * scheduler.get_last_lr()[0] * torch.randn_like(x_trial)
                    # Project into image space
                    if self.config['boxed']:
                        x_trial.data = torch.clamp(x_trial, min=0, max=1)


                    if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        else:
                            raise ValueError()

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            self.generator.zero_grad()

#             dm, ds = self.mean_std
#             torch.clamp(self.generator(x_trial), min=(1 - dm) / ds, max=-dm / ds)

            loss = self.loss_fn(self.model(self.generator(x_trial)), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

            if self.config['EOT_DP'] > 0:
                eot_steps = 10
                rec_losses = torch.zeros((eot_steps,), **self.setup)
                for i in range(eot_steps):
                    gradient_dp =  [grad + torch.normal(torch.zeros_like(grad), self.config['EOT_DP']*torch.ones_like(grad)) for grad in gradient]
                    rec_losses[i] = reconstruction_costs([gradient_dp], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])
                indx = torch.argmin(rec_losses)
                rec_loss = rec_losses[indx]
            else:
                if self.config['EOT_C'] > 0:
                    gradient = [torch.clamp(grad, -self.config['EOT_C'], self.config['EOT_C']) for grad in gradient]
                if self.config['perturb_rep'] != 0:
                    mask = input_gradient[self.config['perturb_rep']][0]!=0
                    gradient = [grad for grad in gradient]
                    gradient[self.config['perturb_rep']] = gradient[self.config['perturb_rep']] * mask
                rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(self.generator(x_trial))

            if self.config['KL'] > 0:
                KLD = -0.5 * torch.sum(1 + torch.log(torch.std(x_trial.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(x_trial.squeeze(), axis=-1).pow(2) - torch.std(x_trial.squeeze(), unbiased=False, axis=-1).pow(2))
                rec_loss += self.config['KL'] * KLD
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] in ['loss', 'last']:
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(self.generator(x_trial)), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(self.generator(x_trial))
            if self.config['KL'] > 0:
                # see Appendix B from VAE paper:
                # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                # https://arxiv.org/abs/1312.6114
                KLD = -0.5 * torch.sum(1 + torch.log(torch.std(x_trial.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(x_trial.squeeze(), axis=-1).pow(2) - torch.std(x_trial.squeeze(), unbiased=False, axis=-1).pow(2))
                rec_loss += self.config['KL'] * KLD
            if self.config['l2_penalty'] > 0:
                l2_penalty = (torch.sum(x_trial.squeeze().pow(2)) - x_trial.squeeze().ndim).pow(2)
                rec_loss += self.config['l2_penalty'] * l2_penalty
            return rec_loss
        elif self.config['scoring_choice'] == 'tv':
            return TV(self.generator(x_trial))
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(self.generator(x_trial))
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(self.generator(x_optimal)).softmax(dim=1)
        loss = self.loss_fn(self.model(self.generator(x_optimal)), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats


class ConditionalGeneratorBasedGradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, generator, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.generator = generator
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        self.num_classes = 10

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.iDLG = True

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), label_pad=True, g_args=None, dryrun=False, eval=True, tol=None):
        """
        Reconstruct image from gradient.
        @param:
        - label_pad: pad label to be of size [batch_size, class_num, 1, 1] if label_pad=True
        else [batch_size, class_num]
        - g_args: arguments for GAN model

        """
        start_time = time.time()

        self.g_args = g_args

        if eval:
            self.model.eval()
            self.generator.eval()


        stats = defaultdict(list)
        x = self._init_images(img_shape)


        scores = torch.zeros(self.config['restarts'])

        # Infer Label information
        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # sum works better than column-min under DP setting
                last_weight_min = torch.argsort(torch.sum(input_data[-2], dim=-1), dim=-1)[:self.num_images]
                labels = last_weight_min.detach().reshape((-1,)).requires_grad_(False)
                self.reconstruct_label = False

            print("Inferred Labels: {}".format(labels))
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        if label_pad:
            self.infer_y = infer_y = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)[:, :, None, None].to(**self.setup)
        else:
            self.infer_y = infer_y = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).to(**self.setup)

        try:
            for trial in range(self.config['restarts']):
                x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass


        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        elif self.config['scoring_choice'] == 'last':
            x_optimal = x[-1]
            stats['opt'] = scores[-1].item()
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        if self.g_args is None:
            return x_optimal.detach(), self.generator(x_optimal, self.infer_y), stats
        else:
            return x_optimal.detach(), self.generator(x_optimal, self.infer_y, **self.g_args), stats
            # return x_optimal.detach(), stats

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        x_trial.requires_grad = True

        if self.reconstruct_label:
            if self.g_args is None:
                output_test = self.model(self.generator(x_trial, self.infer_y))
            else:
                output_test = self.model(self.generator(x_trial, self.infer_y, **self.g_args))
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels])
            else:
                raise ValueError()
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial])
            else:
                raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    if self.config['optim_noise'] > 0:
                        x_trial.data = x_trial.data + self.config['optim_noise'] * scheduler.get_last_lr()[0] * torch.randn_like(x_trial)
                    # Project into image space
                    if self.config['boxed']:
#                         x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)
                        x_trial.data = torch.clamp(x_trial, min=0, max=1)


                    if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        else:
                            raise ValueError()

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):

        def closure():

            optimizer.zero_grad()
            self.model.zero_grad()
            self.generator.zero_grad()

            if self.g_args is None:
                loss = self.loss_fn(self.model(self.generator(x_trial, self.infer_y)), label)
            else:

                loss = self.loss_fn(self.model(self.generator(x_trial, self.infer_y, **self.g_args)), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

            if self.config['EOT_C'] > 0:
                gradient = [torch.clamp(grad, -self.config['EOT_C'], self.config['EOT_C']) for grad in gradient]
            if self.config['EOT_DP'] > 0:

                gradient =  [grad + torch.normal(torch.zeros_like(grad), self.config['EOT_DP']*torch.ones_like(grad)) for grad in gradient]

            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                if self.g_args is None:
                    rec_loss += self.config['total_variation'] * TV(self.generator(x_trial, self.infer_y))
                else:
                    rec_loss += self.config['total_variation'] * TV(self.generator(x_trial, self.infer_y, **self.g_args))

            if self.config['KL'] > 0:
                # see Appendix B from VAE paper:
                # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                # https://arxiv.org/abs/1312.6114
                KLD = -0.5 * torch.sum(1 + torch.log(torch.std(x_trial.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(x_trial.squeeze(), axis=-1).pow(2) - torch.std(x_trial.squeeze(), unbiased=False, axis=-1).pow(2))
                rec_loss += (self.config['KL']/self.num_images) * KLD
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):

        if self.config['scoring_choice'] in ['loss', 'last']:
            self.model.zero_grad()
            x_trial.grad = None
            if self.g_args is None:
                loss = self.loss_fn(self.model(self.generator(x_trial, self.infer_y)), label)
            else:
                loss = self.loss_fn(self.model(self.generator(x_trial, self.infer_y, **self.g_args)), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
            if self.config['total_variation'] > 0:
                if self.g_args is None:
                    rec_loss += self.config['total_variation'] * TV(self.generator(x_trial, self.infer_y))
                else:
                    rec_loss += self.config['total_variation'] * TV(self.generator(x_trial, self.infer_y, **self.g_args))
            if self.config['KL'] > 0:
                # see Appendix B from VAE paper:
                # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                # https://arxiv.org/abs/1312.6114
                KLD = -0.5 * torch.sum(1 + torch.log(torch.std(x_trial.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(x_trial.squeeze(), axis=-1).pow(2) - torch.std(x_trial.squeeze(), unbiased=False, axis=-1).pow(2))
                rec_loss += self.config['KL'] * KLD
            return rec_loss
        elif self.config['scoring_choice'] == 'tv':
            if self.g_args is None:
                return TV(self.generator(x_trial, self.infer_y))
            else:
                return TV(self.generator(x_trial, self.infer_y, **self.g_args))
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            if self.g_args is None:
                return self.inception(self.generator(x_trial, self.infer_y))
            else:
                return self.inception(self.generator(x_trial, self.infer_y, **self.g_args))
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            if self.g_args is None:
                labels = self.model(self.generator(x_optimal, self.infer_y)).softmax(dim=1)
            else:
                labels = self.model(self.generator(x_optimal, self.infer_y, **self.g_args)).softmax(dim=1)
        if self.g_args is None:
            loss = self.loss_fn(self.model(self.generator(x_optimal, self.infer_y)), labels)
        else:
            loss = self.loss_fn(self.model(self.generator(x_optimal, self.infer_y, **self.g_args)), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats



class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config=DEFAULT_CONFIG, num_images=1, use_updates=True, batch_size=0):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size

    def _gradient_closure(self, optimizer, x_trial, input_parameters, labels):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr,
                                    use_updates=self.use_updates,
                                    batch_size=self.batch_size)
            rec_loss = reconstruction_costs([parameters], input_parameters,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_parameters, labels):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            return reconstruction_costs([parameters], input_parameters,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)


def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True, batch_size=0):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            idx = i % (inputs.shape[0] // batch_size)
            outputs = patched_model(inputs[idx * batch_size:(idx + 1) * batch_size], patched_model.parameters)
            labels_ = labels[idx * batch_size:(idx + 1) * batch_size]
        loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)
