#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthtorch.learn.predict

routines specific to prediction

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb 26, 2018
"""

__all__ = ['Predictor']

from typing import Tuple

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Predictor:

    def __init__(self, model:torch.nn.Module, patch_size:Tuple[int], batch_size:int, device:torch.device,
                 axis:int=0, dim:int=3, mean:Tuple[float]=None, std:Tuple[float]=None, tfm_x:bool=True, tfm_y:bool=False):
        self.model = model
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.device = device
        self.axis = axis
        self.n_output = model.n_output
        self.dim = dim
        self.mean = mean
        self.std = std
        self.tfm_x = tfm_x
        self.tfm_y = tfm_y

    def predict(self, img:np.ndarray, nsyn:int=1, calc_var:bool=False) -> np.ndarray:
        """ picks and runs the correct prediction routine based on input info """
        if self.tfm_x and self.mean is not None and self.std is not None:
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                img[i] = (img[i] - m) / s
        if self.patch_size is not None and self.dim == 3:
            out_img = self.patch_3d_predict(img, nsyn, calc_var)
        elif self.dim == 3:
            out_img = self.whole_3d_predict(img, nsyn, calc_var)
        else:
            out_img = self.slice_predict(img, nsyn, calc_var)
        if self.tfm_y and self.mean is not None and self.std is not None:
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                out_img[i] = (out_img[i] * s) + m
        return out_img

    def whole_3d_predict(self, img:np.ndarray, nsyn:int=1, calc_var:bool=False) -> np.ndarray:
        """ 3d whole-image-based prediction """
        if img.ndim == 3: img = img[np.newaxis, ...]
        out_img = np.zeros((nsyn, self.n_output) + img.shape[1:])
        test_img = torch.from_numpy(img).to(self.device)[None, ...]  # add empty batch dimension
        for j in range(nsyn):
            out_img[j] = self._fwd(test_img)[0]  # remove empty batch dimension
        out_img = np.mean(out_img, axis=0) if not calc_var else np.var(out_img, axis=0)
        return out_img

    def patch_3d_predict(self, img:np.ndarray, nsyn:int=1, calc_var:bool=False) -> np.ndarray:
        """ 3d patch-by-patch based prediction """
        if img.ndim == 3: img = img[np.newaxis, ...]
        # pad image to get full image coverage in patch-based processing
        sz, psz = img.shape, self.patch_size
        pad = [int((ps // 2) * np.ceil(sz[i]/(ps // 2)) - sz[i]) for i, ps in enumerate(psz, 1)]
        img = np.asarray(F.pad(torch.from_numpy(img[np.newaxis,...]),
                               [0, pad[2] if psz[2] != sz[3] else 0,
                                0, pad[1] if psz[1] != sz[2] else 0,
                                0, pad[0] if psz[0] != sz[1] else 0], mode='replicate')[0])
        out_img = np.zeros((self.n_output,) + img.shape[1:])
        count_mtx = np.zeros(img.shape[1:])
        x, y, z = self._get_overlapping_3d_idxs(self.patch_size, img)
        # The below for-loop handles collecting overlapping patches and putting
        # them into a batch format that pytorch models expect (i.e., [N,C,H,W,D])
        # and running the batch through the network (storing the results in out_img).
        j = 0
        batch, idxs = [], []
        for i, (xx, yy, zz) in enumerate(zip(x, y, z)):
            if j < self.batch_size:
                idxs.append((xx, yy, zz))
                batch.append(img[:, xx, yy, zz])
                j += 1
            else:
                self.__batch_3d_proc(batch, idxs, nsyn, out_img, count_mtx)
                # restart new batch
                batch, idxs = [img[:, xx, yy, zz]], [(xx, yy, zz)]
                j = 1
        if np.any(count_mtx == 0):
            self.__batch_3d_proc(batch, idxs, nsyn, out_img, count_mtx)
        if np.any(count_mtx == 0):
            logger.warning(f'Part of the synthesized image not covered ({np.sum(count_mtx == 0)} voxels)')
            count_mtx[count_mtx == 0] = 1  # avoid division by zero
        out_img /= count_mtx
        out_img = out_img[:,:sz[1],:sz[2],:sz[3]]
        return out_img

    def __batch_3d_proc(self, batch, idxs, nsyn, out_img, count_mtx):
        bs = len(batch)
        batch = torch.from_numpy(np.stack(batch)).to(self.device)
        predicted = np.zeros((bs,self.n_output,) + batch.shape[2:])
        for _ in range(nsyn):
            predicted += self._fwd(batch) / nsyn
        for ii, (bx, by, bz) in enumerate(idxs):
            out_img[:, bx, by, bz] += predicted[ii, ...]
            count_mtx[bx, by, bz] += 1

    def slice_predict(self, img:np.ndarray, nsyn:int=1, calc_var:bool=False) -> np.ndarray:
        """ slice-by-slice based prediction """
        if img.ndim == 3: img = img[np.newaxis, ...]  # add batch dimension if empty
        out_img = np.zeros((nsyn, self.n_output) + img.shape[1:])
        num_batches = math.floor(img.shape[self.axis + 1] / self.batch_size)  # add one to axis to ignore channel dim
        if img.shape[self.axis + 1] / self.batch_size != num_batches:
            lbi = int(num_batches * self.batch_size)  # last batch index
            num_batches += 1
            lbs = img.shape[self.axis + 1] - lbi  # last batch size
        else:
            lbi = None
        for i in range(num_batches if lbi is None else num_batches - 1):
            logger.info(f'Starting batch ({i + 1}/{num_batches})')
            self._batch2d(img, out_img, i * self.batch_size, nsyn)
        if lbi is not None:
            logger.info(f'Starting batch ({num_batches}/{num_batches})')
            self._batch2d(img, out_img, lbi, nsyn, lbs)
        out_img = np.mean(out_img, axis=0) if not calc_var else np.var(out_img, axis=0)
        return out_img

    def img_predict(self, img:np.ndarray, nsyn:int=1, calc_var:bool=False) -> np.ndarray:
        if img.ndim == 3: img = img[np.newaxis, ...]
        out_img = np.zeros((nsyn, self.n_output) + img.shape[1:])
        for i in range(nsyn):
            out_img[i, ...] = self._fwd(torch.from_numpy(img).to(self.device))
        out_img = np.mean(out_img, axis=0) if not calc_var else np.var(out_img, axis=0)
        return out_img.squeeze()

    def png_predict(self, img:np.ndarray, nsyn:int=1, calc_var:bool=False,
                    scale:bool=False) -> np.ndarray:
        out = self.img_predict(img, nsyn, calc_var)
        if scale:
            a = (img.max() - img.min()) / (out.max() - out.min() + np.finfo(np.float32).eps)
            b = img.min() - (a * out.min())
            out = a * out + b
        out_img = np.asarray(np.around(out), dtype=np.uint8)
        return out_img

    def _fwd(self, img):
        with torch.no_grad():
            out = self.model.predict(img).cpu().detach().numpy()
        return out

    def _get_overlapping_3d_idxs(self, psz, img):
        indices = [self.__unfold(torch.from_numpy(idxs), psz, img) for idxs in np.indices(img.shape[1:])]
        x, y, z = [np.asarray(idxs.contiguous().view(-1, psz[0], psz[1], psz[2])) for idxs in indices]
        return x, y, z

    def __unfold(self, idxs, psz, img):
        return idxs.unfold(0, psz[0], psz[0]//(2 if psz[0] != img.shape[1] else 1))\
                   .unfold(1, psz[1], psz[1]//(2 if psz[1] != img.shape[2] else 1))\
                   .unfold(2, psz[2], psz[2]//(2 if psz[2] != img.shape[3] else 1))

    def _batch2d(self, img, out_img, i, nsyn, bs=None):
        bs = bs or self.batch_size
        s = np.transpose(img[:,i:i+bs,:,:],[1,0,2,3]) if self.axis == 0 else \
            np.transpose(img[:,:,i:i+bs,:],[2,0,1,3]) if self.axis == 1 else \
            np.transpose(img[:,:,:,i:i+bs],[3,0,1,2])
        img_b = torch.from_numpy(s).to(self.device)
        for j in range(nsyn):
            x = self._fwd(img_b)
            if self.axis == 0:
                out_img[j,:,i:i+bs,:,:] = np.transpose(x, [1,0,2,3])
            elif self.axis == 1:
                out_img[j,:,:,i:i+bs,:] = np.transpose(x, [1,2,0,3])
            else:
                out_img[j,:,:,:,i:i+bs] = np.transpose(x, [1,2,3,0])
