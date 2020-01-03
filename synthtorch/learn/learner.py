#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.learn.learner

train functions for synthtorch neural networks

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 25, 2018
"""

__all__ = ['get_data_augmentation',
           'get_dataloader',
           'get_device',
           'get_model',
           'Learner']

from dataclasses import dataclass
from typing import List, Tuple, Union

import logging
import os
import random

import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose

from niftidataset import MultimodalNiftiDataset, MultimodalImageDataset, split_filename
import niftidataset.transforms as niftitfms

from ..errors import SynthtorchError
from ..plot.loss import plot_loss
from .predict import Predictor
from ..util.config import ExperimentConfig
from ..util.helper import get_optim, init_weights

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None

try:
    from apex import amp
except (ImportError, ModuleNotFoundError):
    amp = None

logger = logging.getLogger(__name__)


class Learner:

    def __init__(self, model, device=None, train_loader=None, valid_loader=None, optimizer=None,
                 predictor=None, config=None):
        self.model = model
        self.model_name = model.__class__.__name__.lower()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.config = config
        self.record = None
        self.use_fp16 = False

    @classmethod
    def train_setup(cls, config:Union[str,ExperimentConfig]):
        if isinstance(config,str):
            config = ExperimentConfig.load_json(config)
        if isinstance(config.kernel_size, int):
            config.kernel_size = tuple([config.kernel_size for _ in range(config.dim)])
        device, use_cuda = get_device(config.disable_cuda)
        if config.color: config.n_input, config.n_output = config.n_input * 3, config.n_output * 3
        model = get_model(config, True, False)
        if config.color: config.n_input, config.n_output = config.n_input // 3, config.n_output // 3
        logger.debug(model)
        logger.info(f'Number of trainable parameters in model: {num_params(model)}')
        load_chkpt = os.path.isfile(config.trained_model)
        checkpoint = torch.load(config.trained_model, map_location=device) if load_chkpt else None
        if load_chkpt:
            logger.info(f"Loading checkpoint: {config.trained_model} (epoch {checkpoint['epoch']})")
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)
        else:
            logger.info(f'Initializing weights with {config.init}')
            init_weights(model, config.init, config.init_gain)
        if use_cuda: model.cuda(device=device)
        train_loader, valid_loader = get_dataloader(config)
        if config.lr_scheduler is None: logger.info(f'LR: {config.learning_rate:.2e}')
        def gopt(name, mp, **kwargs):
            return get_optim(name)(mp, lr=config.learning_rate, weight_decay=config.weight_decay, **kwargs)
        try:
            optimizer = gopt(config.optimizer, model.parameters(), betas=config.betas)
        except TypeError:
            try:
                optimizer = gopt(config.optimizer, model.parameters(), momentum=config.betas[0])
            except TypeError:
                optimizer = gopt(config.optimizer, model.parameters())
        if load_chkpt and not config.no_load_opt:
            optimizer.load_state_dict(checkpoint['optimizer'])
        model.train()
        if config.freeze: model.freeze()
        predictor = Predictor(model, config.patch_size, config.batch_size, device, config.sample_axis,
                              config.dim, config.mean, config.std, config.tfm_x, config.tfm_y)
        return cls(model, device, train_loader, valid_loader, optimizer, predictor, config)

    @classmethod
    def predict_setup(cls, config:Union[str,ExperimentConfig]):
        if isinstance(config,str):
            config = ExperimentConfig.load_json(config)
        if isinstance(config.kernel_size, int):
            config.kernel_size = tuple([config.kernel_size for _ in range(config.dim)])
        device, use_cuda = get_device(config.disable_cuda)
        nsyn = config.monte_carlo or 1
        if config.color: config.n_input, config.n_output = config.n_input * 3, config.n_output * 3
        model = get_model(config, nsyn > 1 and config.dropout_prob > 0, True)
        logger.debug(model)
        checkpoint = torch.load(config.trained_model, map_location=device)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        if use_cuda: model.cuda(device=device)
        model.eval()
        predictor = Predictor(model, config.patch_size, config.batch_size, device, config.sample_axis,
                              config.dim, config.mean, config.std, config.tfm_x, config.tfm_y)
        return cls(model, device, predictor=predictor, config=config)

    def fit(self, n_epochs, clip:float=None, checkpoint:int=None, trained_model:str=None):
        """ training loop for neural network """
        self.model.train()
        use_tb = self.config.tensorboard and SummaryWriter is not None
        if use_tb: writer = SummaryWriter()
        use_valid = self.valid_loader is not None
        use_scheduler = hasattr(self, 'scheduler')
        use_restarts = self.config.lr_scheduler == 'cosinerestarts'
        train_losses, valid_losses = [], []
        n_batches = len(self.train_loader)
        for t in range(1, n_epochs + 1):
            # training
            t_losses = []
            if use_valid: self.model.train(True)
            for i, (src, tgt) in enumerate(self.train_loader):
                src, tgt = src.to(self.device), tgt.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(src)
                loss = self._criterion(out, tgt)
                t_losses.append(loss.item())
                if self.use_fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if clip is not None: nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
                if use_scheduler: self.scheduler.step(((t-1)+(i/n_batches)) if use_restarts else None)
                if use_tb:
                    if i % 20 == 0: writer.add_scalar('Loss/train', loss.item(), ((t-1)*n_batches)+i)

                del loss  # save memory by removing ref to gradient tree
            train_losses.append(t_losses)

            if checkpoint is not None:
                if t % checkpoint == 0:
                    path, base, ext = split_filename(trained_model)
                    fn = os.path.join(path, base + f'_chk_{t}' + ext)
                    self.save(fn, t)

            # validation
            v_losses = []
            if use_valid:
                self.model.train(False)
                with torch.no_grad():
                    for i, (src, tgt) in enumerate(self.valid_loader):
                        src, tgt = src.to(self.device), tgt.to(self.device)
                        out = self.model(src)
                        loss = self._criterion(out, tgt)
                        if use_tb:
                            if i % 20 == 0: writer.add_scalar('Loss/valid', loss.item(), ((t-1)*n_batches)+i)
                            do_plot = i == 0 and ((t - 1) % 5) == 0
                            if do_plot and self.model.dim == 2:
                                writer.add_images('source', src[:8], t, dataformats='NCHW')
                                outimg = out[0][:8] if isinstance(out, tuple) else out[:8]
                                if self.config.color: outimg = torch.round(outimg)
                                writer.add_images('target', outimg, t, dataformats='NCHW')
                            if do_plot: self._histogram_weights(writer, t)
                        v_losses.append(loss.item())
                    valid_losses.append(v_losses)

            if not np.all(np.isfinite(t_losses)): raise SynthtorchError('NaN or Inf in training loss, cannot recover. Exiting.')
            if logger is not None:
                log = f'Epoch: {t} - Training Loss: {np.mean(t_losses):.2e}'
                if use_valid: log += f', Validation Loss: {np.mean(v_losses):.2e}'
                if use_scheduler: log += f', LR: {self.scheduler.get_lr()[0]:.2e}'
                logger.info(log)

        self.record = Record(train_losses, valid_losses)
        if use_tb: writer.close()

    def predict(self, fn:str, nsyn:int=1, calc_var:bool=False):
        self.model.eval()
        f = fn[0].lower()
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            img_nib = nib.load(fn[0])
            img = np.stack([np.asarray(nib.load(f).get_data(), dtype=np.float32) for f in fn])
            out = self.predictor.predict(img, nsyn, calc_var)
            out_img = [nib.Nifti1Image(o, img_nib.affine, img_nib.header) for o in out]
        elif f.split('.')[-1] in ('tif', 'tiff', 'png', 'jpg', 'jpeg'):
            out_img = self._img_predict(fn, nsyn, calc_var)
        else:
            raise SynthtorchError(f'File: {fn[0]}, not supported.')
        return out_img

    def _img_predict(self, fn, nsyn, calc_var):
        img = np.stack([np.asarray(Image.open(f), dtype=np.float32) for f in fn])
        if self.config.color: img = img.transpose((0,3,1,2))
        out = self.predictor.img_predict(img, nsyn, calc_var)
        if self.config.color: 
            out = out.transpose((1,2,0))  # only support one color image as output
            out = [np.around(out[...,0:3]).astype(np.uint8)] + [out[...,i] for i in range(3,out.shape[-1])] \
                  if self.config.nn_arch not in ('nconv','unet','densenet') else \
                  np.around(out[None,...]).astype(np.uint8)
        return [Image.fromarray(o) for o in out]

    def _criterion(self, out, tgt):
        """ helper function to handle multiple outputs in model evaluation """
        c = self.model.module.criterion if isinstance(self.model,nn.DataParallel) else self.model.criterion
        return c(out, tgt)

    def fp16(self):
        """ import and initialize mixed precision training package """
        if amp is not None:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
            self.use_fp16 = True
        else:
            logger.info('Mixed precision training (i.e., the package `apex`) not available.')

    def multigpu(self):
        """ put the model on the GPU if available and desired """
        n_gpus = torch.cuda.device_count()
        if n_gpus <= 1:
            logger.warning('Multi-GPU functionality is not available on your system.')
        else:
            logger.info(f'Enabling use of {n_gpus} gpus')
            self.model = torch.nn.DataParallel(self.model)

    def lr_scheduler(self, n_epochs, lr_scheduler='cyclic', restart_period=None, t_mult=None,
                     num_cycles=1, cycle_mode='triangular', momentum_range=(0.85,0.95), div_factor=25, pct_start=0.3, **kwargs):
        lr = self.config.learning_rate
        if lr_scheduler == 'cyclic':
            logger.info(f'Enabling cyclic LR scheduler with {num_cycles} cycle(s)')
            ss = int((n_epochs * len(self.train_loader)) / num_cycles)
            ssu = int(pct_start * ss)
            ssd = ss - ssu
            cycle_momentum = self.config.optimizer in ('sgd','sgdw','nsgd','nsgdw','rmsprop')
            momentum_kwargs = {'cycle_momentum': cycle_momentum}
            if not cycle_momentum and momentum_range is not None:
                logger.warning(f'{self.config.optimizer} not compatible with momentum cycling, disabling.')
            elif momentum_range is not None:
                momentum_kwargs.update({'base_momentum': momentum_range[0], 'max_momentum': momentum_range[1]})
            self.scheduler = CyclicLR(self.optimizer, lr/div_factor, lr, step_size_up=ssu, step_size_down=ssd,
                                      mode=cycle_mode, **momentum_kwargs)
        elif lr_scheduler == 'cosinerestarts':
            logger.info('Enabling cosine annealing with restarts LR scheduler')
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, restart_period, T_mult=t_mult, eta_min=lr/div_factor)
        else:
            raise SynthtorchError(f'Invalid type {type} for scheduler.')
        logger.info(f'Max LR: {lr:.2e}, Min LR: {lr/div_factor:.2e}')

    def load(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        logger.info(f"Loaded checkpoint: {fn} (epoch {checkpoint['epoch']})")
        if 'amp' in checkpoint.keys():
            amp.initialize(self.model, self.optimizer, opt_level='O1')
            amp.load_state_dict(checkpoint['amp'])
        self.model.load_state_dict(checkpoint['model']).to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save(self, fn, epoch=0):
        """ save a model, an optimizer state and the epoch number to a file """
        model = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        state = {'epoch': epoch, 'model': model, 'optimizer': self.optimizer.state_dict()}
        if self.use_fp16: state['amp'] = amp.state_dict()
        torch.save(state, fn)

    def _histogram_weights(self, writer, epoch):
        """ write histogram of weights to tensorboard """
        for (name, values) in self.model.named_parameters():
            writer.add_histogram(tag='weights/'+name, values=values.clone().detach().cpu(), global_step=epoch)


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(config:ExperimentConfig, enable_dropout:bool=True, inplace:bool=False):
    """
    instantiate a model based on an ExperimentConfig class instance

    Args:
        config (ExperimentConfig): instance of the ExperimentConfig class
        enable_dropout (bool): enable dropout in the model (usually for training)

    Returns:
        model: instance of one of the available models in the synthtorch package
    """
    if config.nn_arch == 'nconv':
        from ..models.nconvnet import SimpleConvNet
        logger.warning('The nconv network is for basic testing.')
        model = SimpleConvNet(**config)
    elif config.nn_arch == 'unet':
        from ..models.unet import Unet
        model = Unet(enable_dropout=enable_dropout, inplace=inplace, **config)
    elif config.nn_arch == 'vae':
        from ..models.vae import VAE
        model = VAE(**config)
    elif config.nn_arch == 'densenet':
        from ..models.densenet import DenseNet
        model = DenseNet(**config)
    elif config.nn_arch == 'ordnet':
        try:
            from annom.models import OrdNet
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the OrdNet without the annom toolbox.')
        model = OrdNet(enable_dropout=enable_dropout, inplace=inplace, **config)
    elif config.nn_arch == 'hotnet':
        try:
            from annom.models import HotNet
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the HotNet without the annom toolbox.')
        model = HotNet(inplace=inplace, **config)
    elif config.nn_arch == 'burnnet':
        try:
            from annom.models import BurnNet
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the BurnNet without the annom toolbox.')
        model = BurnNet(inplace=inplace, **config)
    elif config.nn_arch == 'burn2net':
        try:
            from annom.models import Burn2Net
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the Burn2Net without the annom toolbox.')
        model = Burn2Net(inplace=inplace, **config)
    elif config.nn_arch == 'burn2netp12':
        try:
            from annom.models import Burn2NetP12
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the Burn2NetP12 without the annom toolbox.')
        model = Burn2NetP12(inplace=inplace, **config)
    elif config.nn_arch == 'burn2netp21':
        try:
            from annom.models import Burn2NetP21
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the Burn2NetP21 without the annom toolbox.')
        model = Burn2NetP21(inplace=inplace, **config)
    elif config.nn_arch == 'unburnnet':
        try:
            from annom.models import UnburnNet
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the UnburnNet without the annom toolbox.')
        model = UnburnNet(inplace=inplace, **config)
    elif config.nn_arch == 'unburn2net':
        try:
            from annom.models import Unburn2Net
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the Unburn2Net without the annom toolbox.')
        model = Unburn2Net(inplace=inplace, **config)
    elif config.nn_arch == 'lavanet':
        try:
            from annom.models import LavaNet
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the LavaNet without the annom toolbox.')
        model = LavaNet(inplace=inplace, **config)
    elif config.nn_arch == 'lava2net':
        try:
            from annom.models import Lava2Net
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the Lava2Net without the annom toolbox.')
        model = Lava2Net(inplace=inplace, **config)
    elif config.nn_arch == 'lautonet':
        try:
            from annom.models import LAutoNet
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the LAutoNet without the annom toolbox.')
        model = LAutoNet(enable_dropout=enable_dropout, inplace=inplace, **config)
    elif config.nn_arch == 'ocnet1':
        try:
            from annom.models import OCNet1
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the OCNet without the annom toolbox.')
        model = OCNet1(enable_dropout=enable_dropout, inplace=inplace if config.dropout_prob == 0 else False, **config)
    elif config.nn_arch == 'ocnet2':
        try:
            from annom.models import OCNet2
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use the OCNet without the annom toolbox.')
        model = OCNet2(enable_dropout=enable_dropout, inplace=inplace if config.dropout_prob == 0 else False, **config)
    else:
        raise SynthtorchError(f'Invalid NN type: {config.nn_arch}. '
                              f'{{nconv,unet,vae,densenet,ordnet,hotnet,burnnet,burn2netp12,burn2netp21,'
                              f'unburnnet,unburn2net,lavanet,lava2net,lautonet,ocnet1,ocnet2}} '
                              f'are the only supported options.')
    return model


def get_device(disable_cuda=False):
    """ get the device(s) for tensors to be put on """
    cuda_avail = torch.cuda.is_available()
    use_cuda = cuda_avail and not disable_cuda
    if use_cuda: torch.backends.cudnn.benchmark = True
    if not cuda_avail and not disable_cuda: logger.warning('CUDA does not appear to be available on your system.')
    device = torch.device("cuda" if use_cuda else "cpu")
    return device, use_cuda


def get_dataloader(config:ExperimentConfig, tfms:Tuple[List,List]=None):
    """ get the dataloaders for training/validation """
    if config.dim > 1:
        # get data augmentation if not defined
        train_tfms, valid_tfms = get_data_augmentation(config) if tfms is None else tfms

        # check number of jobs requested and CPUs available
        num_cpus = os.cpu_count()
        if num_cpus < config.n_jobs:
            logger.warning(f'Requested more workers than available (n_jobs={config.n_jobs}, # cpus={num_cpus}). '
                           f'Setting n_jobs={num_cpus}.')
            config.n_jobs = num_cpus

        # define dataset and split into training/validation set
        use_nii_ds = config.ext is None or 'nii' in config.ext
        dataset = MultimodalNiftiDataset(config.source_dir, config.target_dir, Compose(train_tfms),
                                         preload=config.preload) if use_nii_ds else \
                  MultimodalImageDataset(config.source_dir, config.target_dir, Compose(train_tfms),
                                         ext='*.' + config.ext, color=config.color, preload=config.preload)
        logger.info(f'Number of training images: {len(dataset)}')

        if config.valid_source_dir is not None and config.valid_target_dir is not None:
            valid_dataset = MultimodalNiftiDataset(config.valid_source_dir, config.valid_target_dir,
                                                   Compose(valid_tfms), preload=config.preload) if use_nii_ds else \
                            MultimodalImageDataset(config.valid_source_dir, config.valid_target_dir, Compose(valid_tfms),
                                                   ext='*.' + config.ext, color=config.color, preload=config.preload)
            logger.info(f'Number of validation images: {len(valid_dataset)}')
            train_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.n_jobs, shuffle=True,
                                      pin_memory=config.pin_memory, worker_init_fn=init_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=config.n_jobs,
                                      pin_memory=config.pin_memory, worker_init_fn=init_fn)
        else:
            # setup training and validation set
            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(config.valid_split * num_train)
            valid_idx = np.random.choice(indices, size=split, replace=False)
            train_idx = list(set(indices) - set(valid_idx))
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            # set up data loader for nifti images
            train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=config.batch_size,
                                      num_workers=config.n_jobs, pin_memory=config.pin_memory, worker_init_fn=init_fn)
            valid_loader = DataLoader(dataset, sampler=valid_sampler, batch_size=config.batch_size,
                                      num_workers=config.n_jobs, pin_memory=config.pin_memory, worker_init_fn=init_fn)
    else:
        try:
            from altdataset import CSVDataset
        except (ImportError, ModuleNotFoundError):
            raise SynthtorchError('Cannot use 1D ConvNet in CLI without the altdataset toolbox.')
        train_dataset, valid_dataset = CSVDataset(config.source_dir[0]), CSVDataset(config.valid_source_dir[0])
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.n_jobs, shuffle=True,
                                  pin_memory=config.pin_memory)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=config.n_jobs,
                                  pin_memory=config.pin_memory)

    return train_loader, valid_loader


def init_fn(worker_id):
    random.seed((torch.initial_seed() + worker_id) % (2**32))
    np.random.seed((torch.initial_seed() + worker_id) % (2**32))


def get_data_augmentation(config:ExperimentConfig):
    """ get all data augmentation transforms for training """
    train_tfms, valid_tfms = [], []

    # add data augmentation if desired
    if config.prob is not None:
        logger.info('Adding data augmentation transforms')
        train_tfms.extend(niftitfms.get_transforms(config.prob, config.tfm_x, config.tfm_y, config.rotate, config.translate,
                                                   config.scale, config.vflip, config.hflip, config.gamma, config.gain,
                                                   config.noise_pwr, config.block, config.threshold, config.dim == 3,
                                                   config.mean, config.std, config.color))
        if config.mean is not None and config.std is not None:
            valid_tfms.extend([niftitfms.ToTensor(config.color),
                               niftitfms.Normalize(config.mean, config.std, config.tfm_x, config.tfm_y, config.dim == 3)])
    else:
        logger.info('No data augmentation will be used')
        train_tfms.append(niftitfms.ToTensor(config.color))
        valid_tfms.append(niftitfms.ToTensor(config.color))

    # control random cropping patch size (or if used at all)
    if (config.ext is None or config.ext == 'nii') and config.patch_size is not None:
        cropper = niftitfms.RandomCrop3D(config.patch_size, config.threshold, config.sample_pct, config.sample_axis) if config.dim == 3 else \
                  niftitfms.RandomCrop2D(config.patch_size, config.sample_axis, config.threshold)
        train_tfms.append(cropper if config.patch_size is not None and config.dim == 3 else \
                          niftitfms.RandomSlice(config.sample_axis))
        valid_tfms.append(cropper if config.patch_size is not None and config.dim == 3 else \
                          niftitfms.RandomSlice(config.sample_axis))
    else:
        if config.patch_size is not None:
            train_tfms.append(niftitfms.RandomCrop(config.patch_size, config.threshold))
            valid_tfms.append(niftitfms.RandomCrop(config.patch_size, config.threshold))

    logger.debug(f'Training transforms: {train_tfms}')
    return train_tfms, valid_tfms


@dataclass
class Record:
    train_loss: List[List[float]]
    valid_loss: List[List[float]]

    def plot_loss(self, fn:str=None, plot_error:bool=False):
        """ plot training and validation losses on the same plot (with or without error bars) """
        ax = plot_loss(self.train_loss, ecolor='darkorchid', label='Train', plot_error=plot_error)
        _ = plot_loss(self.valid_loss, filename=fn, ecolor='firebrick', ax=ax, label='Validation',
                      plot_error=plot_error)

    def write_csv(self, fn:str):
        """ write training and validation losses to a csv file """
        import csv
        head = ['epochs','avg train','std train','avg valid','std valid']
        epochs = list(range(1, len(self.train_loss) + 1))
        avg_tl = [np.mean(losses) for losses in self.train_loss]
        std_tl = [np.std(losses) for losses in self.train_loss]
        avg_vl = [np.mean(losses) for losses in self.valid_loss]
        std_vl = [np.std(losses) for losses in self.valid_loss]
        out = np.vstack([epochs, avg_tl, std_tl, avg_vl, std_vl]).T
        with open(fn, "w") as f:
            wr = csv.writer(f)
            wr.writerow(head)
            wr.writerows(out)
