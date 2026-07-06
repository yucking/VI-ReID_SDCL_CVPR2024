# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
"""
SDCL v35 full-state Stage-2 handoff.

This version intentionally goes back to the stable v4base training logic instead
of adding another pseudo-label filter or auxiliary neighbourhood loss. The
change is in the main training closure: the Stage-1 -> Stage-2 transition can
save and resume the full optimizer/scheduler/RNG/best-score state, so the
Stage-2 run is no longer a model-only restart from 20model_best.pth.tar.

Primary modes:
  full chain: train from epoch 0 through Stage-2 and write stage2_handoff_full_state.pth.tar.
  full-state resume: resume exactly from that handoff with --stage2-resume-full-state.
  model-only stage2: compatibility fallback with --stage2-only --stage2-init.
"""
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
from solver import make_optimizer, WarmupMultiStepLR
from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import cfg
from clustercontrast import datasets
# from clustercontrast import models
from clustercontrast.model_vit_cmrefine import make_model
from torch import einsum
from clustercontrast.models.cm import ClusterMemory,Memory_wise_v3
from clustercontrast.trainers_source_softweight import ClusterContrastTrainer_SDCL as ClusterContrastTrainer_Source
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint,save_checkpoint10
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance,compute_ranked_list,compute_ranked_list_cm
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam,MoreCameraSampler
import os
import torch.utils.data as data
from torch.autograd import Variable
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,Gray
from collections import Counter
from solver.scheduler_factory import create_scheduler
from typing import Tuple, List, Optional
from torch import Tensor
import numbers
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import cv2

import copy
import os.path as osp
import errno
import shutil
import gc
start_epoch = 0
# best-score is used to select model_best.pth.tar
best_score = -1.0
best_tiebreak = -1.0  # Break ties with indoor_mAP + indoor_mINP.
best_epoch = -1
best_stage1_score = -1.0
best_stage1_tiebreak = -1.0
best_stage1_epoch = -1
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
part=1
torch.backends.cudnn.enable =True,
torch.backends.cudnn.benchmark = True




# l2norm = Normalize(2)



def get_data(name, data_dir):
    '''
    Create a dataset object from its registered name and data root.
    '''
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):
    '''
    Create the infrared training loader.
    train_set: sorted training split.
    rmgs_flag: whether multi-gallery sampling is enabled.
    sampler: selected according to the no_cam flag.
    train_loader: DataLoader wrapped by IterLoader.
    '''
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            # sampler = MoreCameraSampler(train_set, num_instances)
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):
    '''
    Create the RGB training loader; optionally returns two augmented RGB views.
    '''
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            # sampler = MoreCameraSampler(train_set, num_instances)
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(args, dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    '''
    Create an evaluation loader with standard normalization.
    '''
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))
    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    '''
    Create a model, move it to GPU, and wrap it with DataParallel.
    '''
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)
    return model


class TestData(data.Dataset):
    '''
    Test dataset wrapper for SYSU evaluation images and labels.
    '''
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def process_query_sysu(data_path, mode = 'all', relabel=False):
    '''
    Build the SYSU query split for the selected all/indoor protocol.
    '''
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    '''
    Build the SYSU gallery split for the selected all/indoor protocol.
    '''
    random.seed(trial)
    
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
    

def fliplr(img):
    '''Horizontally flip a batch of images by reversing the width dimension.'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
def extract_gall_feat(model,gall_loader,ngall):
    '''
    Extract gallery features with original and horizontally flipped images.
    '''
    pool_dim=768*2
    net = model
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc,feat_fc_s = net( input,input, 1)
            feat_fc = torch.cat((feat_fc,feat_fc_s),dim=1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1,feat_fc_1_s = net( flip_input,flip_input, 1)
            feat_fc_1 = torch.cat((feat_fc_1,feat_fc_1_s),dim=1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc


def extract_query_feat(model,query_loader,nquery):
    pool_dim=768*2
    net = model
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc,feat_fc_s = net( input, input,2)
            feat_fc = torch.cat((feat_fc,feat_fc_s),dim=1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1,feat_fc_1_s= net( flip_input,flip_input, 2)
            feat_fc_1 = torch.cat((feat_fc_1,feat_fc_1_s),dim=1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc



def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP

def pairwise_distance(features_q, features_g):
    x = torch.from_numpy(features_q)
    y = torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m.numpy()





class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



def main():
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    if args.stage2_resume_full_state and args.stage2_only:
        parser.error('--stage2-resume-full-state and --stage2-only are mutually exclusive.')
    if args.stage2_init and not args.stage2_only:
        parser.error('--stage2-init is only valid together with --stage2-only.')
    if args.stage2_only and not args.stage2_init:
        parser.error('--stage2-only requires --stage2-init=/path/to/20model_best.pth.tar.')
    if args.stage2_resume_full_state and not osp.isfile(args.stage2_resume_full_state):
        parser.error('Missing full-state Stage-2 handoff: {}'.format(args.stage2_resume_full_state))
    if args.stage2_init and not osp.isfile(args.stage2_init):
        parser.error('Missing model-only Stage-2 checkpoint: {}'.format(args.stage2_init))
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = False
        cudnn.benchmark = True
    log_name = 'sysu_train'
    main_worker_stage2(args,log_name) #add CMA 

def main_worker_stage2(args,log_name):
    
    l2norm = Normalize(2) # L2 normalization helper kept for compatibility.
    ir_batch = int(args.batch_size)
    rgb_batch = int(args.batch_size)
    print('==> SYSU train IR/RGB batch = -b = {} (grad_accum_steps={})'.format(
        args.batch_size, int(args.grad_accum_steps)))

    global start_epoch, best_score, best_tiebreak, best_epoch
    global best_stage1_score, best_stage1_tiebreak, best_stage1_epoch
    best_score = -1.0
    best_tiebreak = -1.0
    best_epoch = -1
    best_stage1_score = -1.0
    best_stage1_tiebreak = -1.0
    best_stage1_epoch = -1

    # Use ./logs/<log_name> only when --logs-dir is not provided.
    # Grid scripts must pass --logs-dir explicitly to avoid overwriting sysu_train.
    if args.logs_dir is None:
        args.logs_dir = osp.abspath(osp.join('./logs', log_name))
    else:
        args.logs_dir = osp.abspath(args.logs_dir)
    print('==> logs_dir (final): {}'.format(args.logs_dir))
    start_time = time.monotonic()

    cudnn.deterministic = False
    cudnn.benchmark = True
    print('==> memsafe_mode: cudnn benchmark enabled; dataloader seeds fixed')
    
    # Redirect stdout/stderr to the run log file.
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, 'r') as cf:
        config_str = "\n" + cf.read()
    print(config_str)
    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('sysu_ir', args.data_dir)
    dataset_rgb = get_data('sysu_rgb', args.data_dir)

    test_loader_ir = get_test_loader(args, dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(args, dataset_rgb, args.height, args.width, args.batch_size, args.workers)

    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0)
    # Move model to GPU and wrap it with DataParallel for multi-GPU training.
    model.cuda()
    model = nn.DataParallel(model)
    # metric-first main: force source trainer backbone for stability.
    trainer_backend = 'source'
    trainer = ClusterContrastTrainer_Source(model)
    trainer.cmlabel = int(args.cmlabel)
    trainer.hm = 0
    trainer.ht = 10
    trainer.cross_modal_mode = getattr(args, 'cross_modal_mode', 'alternating')
    trainer.grad_accum_steps = max(1, int(args.grad_accum_steps))
    trainer.contrast_tau = float(getattr(args, 'temp', 0.05))
    trainer.sample_weight_ir = None
    trainer.sample_weight_rgb = None
    print('==> trainer-backend: {}'.format(trainer_backend))
    print('==> best-select final={} stage1={}'.format(
        getattr(args, 'best_select_mode', 'full'),
        getattr(args, 'stage1_best_select_mode', 'legacy')))
    print('==> innovation: stage2_softweight_tailtrim={} cglf={} tailtrim={}'.format(
        bool(getattr(args, 'enable_stage2_softweight', False)),
        bool(getattr(args, 'enable_cglf', False)),
        bool(getattr(args, 'enable_stage2_tailtrim', False))))
    # Train only parameters that require gradients.
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]

    # Create the SGD optimizer and LR scheduler.
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # Evaluator used for modality-specific validation.
    evaluator = Evaluator(model)

    def capture_rng_state():
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state_all()
        return state

    def restore_rng_state(rng_state):
        if not rng_state:
            print('[FULLSTATE-RESUME] no RNG state found in checkpoint; continuing with current seed state.')
            return
        try:
            if 'python' in rng_state:
                random.setstate(rng_state['python'])
            if 'numpy' in rng_state:
                np.random.set_state(rng_state['numpy'])
            if 'torch' in rng_state:
                torch.set_rng_state(rng_state['torch'])
            if 'cuda' in rng_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state['cuda'])
            print('[FULLSTATE-RESUME] restored Python/NumPy/Torch RNG states.')
        except Exception as exc:
            print('[FULLSTATE-RESUME][WARN] failed to restore RNG state: {}'.format(exc))

    def serializable_args_snapshot():
        snapshot = {}
        for key, value in vars(args).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                snapshot[key] = value
            elif isinstance(value, (list, tuple)):
                snapshot[key] = list(value)
        return snapshot

    def build_full_state_payload(epoch, next_epoch, source_stage1_checkpoint=''):
        return {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'rng_state': capture_rng_state(),
            'epoch': int(epoch),
            'next_epoch': int(next_epoch),
            'start_epoch': int(next_epoch),
            'cmlabel': int(trainer.cmlabel),
            'best_score': float(best_score),
            'best_tiebreak': float(best_tiebreak),
            'best_epoch': int(best_epoch),
            'best_stage1_score': float(best_stage1_score),
            'best_stage1_tiebreak': float(best_stage1_tiebreak),
            'best_stage1_epoch': int(best_stage1_epoch),
            'source_stage1_checkpoint': source_stage1_checkpoint,
            'source_logs_dir': args.logs_dir,
            'source_model_best_path': osp.join(args.logs_dir, 'model_best.pth.tar'),
            'source_20model_best_path': osp.join(args.logs_dir, '20model_best.pth.tar'),
            'resume_kind': 'v35_full_state_stage2_handoff',
            'args_snapshot': serializable_args_snapshot(),
        }

    def seed_model_best_for_resume(resume_checkpoint):
        dst_model_best = osp.join(args.logs_dir, 'model_best.pth.tar')
        if osp.isfile(dst_model_best):
            return
        src_model_best = resume_checkpoint.get('source_model_best_path', '')
        if src_model_best and osp.isfile(src_model_best):
            shutil.copy(src_model_best, dst_model_best)
            print('[FULLSTATE-RESUME] copied source model_best into new log dir: {}'.format(src_model_best))
            return
        print('[FULLSTATE-RESUME][WARN] source model_best is unavailable; seeding model_best with the handoff model. '
              'Use the original log dir or keep source_model_best_path available for exact pre-Stage2 best fallback.')
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': int(resume_checkpoint.get('next_epoch', trainer.cmlabel)),
            'best_score': float(resume_checkpoint.get('best_score', best_score)),
            'fullstate_seed_model_best': True,
        }, True, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

    loop_start_epoch = 0
    resume_full_state = bool(getattr(args, 'stage2_resume_full_state', ''))
    stage2_model_only = bool(getattr(args, 'stage2_only', False))
    if resume_full_state:
        resume_checkpoint = load_checkpoint(args.stage2_resume_full_state)
        model.load_state_dict(resume_checkpoint['state_dict'])
        if 'optimizer' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
        else:
            print('[FULLSTATE-RESUME][WARN] optimizer state missing; continuing with fresh optimizer.')
        if 'lr_scheduler' in resume_checkpoint:
            lr_scheduler.load_state_dict(resume_checkpoint['lr_scheduler'])
        else:
            print('[FULLSTATE-RESUME][WARN] scheduler state missing; continuing with fresh scheduler.')
        if not getattr(args, 'no_resume_rng', False):
            restore_rng_state(resume_checkpoint.get('rng_state'))
        else:
            print('[FULLSTATE-RESUME] RNG restore disabled by --no-resume-rng.')
        best_score = float(resume_checkpoint.get('best_score', best_score))
        best_tiebreak = float(resume_checkpoint.get('best_tiebreak', best_tiebreak))
        best_epoch = int(resume_checkpoint.get('best_epoch', best_epoch))
        best_stage1_score = float(resume_checkpoint.get('best_stage1_score', best_stage1_score))
        best_stage1_tiebreak = float(resume_checkpoint.get('best_stage1_tiebreak', best_stage1_tiebreak))
        best_stage1_epoch = int(resume_checkpoint.get('best_stage1_epoch', best_stage1_epoch))
        loop_start_epoch = int(resume_checkpoint.get(
            'next_epoch',
            resume_checkpoint.get('start_epoch', resume_checkpoint.get('epoch', trainer.cmlabel)),
        ))
        seed_model_best_for_resume(resume_checkpoint)
        print('[FULLSTATE-RESUME] start_epoch={} cmlabel={} best_epoch={} best_score={:.4f}'.format(
            loop_start_epoch, trainer.cmlabel, best_epoch, best_score))
    elif stage2_model_only:
        loop_start_epoch = int(trainer.cmlabel)
        print('[MODEL-ONLY-STAGE2] start_epoch={} from {}'.format(loop_start_epoch, args.stage2_init))
    if loop_start_epoch >= args.epochs:
        raise ValueError('loop_start_epoch={} must be smaller than --epochs={}'.format(loop_start_epoch, args.epochs))

    @torch.no_grad()
    def generate_cluster_features(labels, features):
        # Store feature lists grouped by pseudo label.
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels): # Skip outliers and collect features by pseudo label.
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())  # Average features for each pseudo-label center.
        ]
        # Stack all cluster centers and return them.
        centers = torch.stack(centers, dim=0)
        return centers

    @torch.no_grad()
    def build_stage2_rgb_soft_weight(confidence, min_weight=0.80, power=1.0):
        conf = confidence.float().clamp(0.0, 1.0)
        power = max(float(power), 1e-6)
        conf = conf.pow(power)
        weights = float(min_weight) + (1.0 - float(min_weight)) * conf
        return weights.clamp(float(min_weight), 1.0)

    # Color jitter augmentation for RGB training images.
    color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    # Standard ImageNet normalization.
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    height=args.height
    width=args.width
    train_transformer_rgb = T.Compose([
    color_aug,
    T.Resize((height, width)),#, interpolation=3
    T.Pad(10),
    T.RandomCrop((height, width)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    normalizer,
    ChannelRandomErasing(probability = 0.5)
    ])

    train_transformer_rgb1 = T.Compose([
    color_aug,
    T.Resize((height, width)),
    T.Pad(10),
    T.RandomCrop((height, width)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    normalizer,
    ChannelRandomErasing(probability = 0.5),
    ChannelExchange(gray = 2)
    ])

    transform_thermal = T.Compose( [
        color_aug,
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelAdapGray(probability =0.5)
        ])
    transform_thermal1 = T.Compose( [
        color_aug,
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelAdapGray(probability =0.5)])

    rgb_cluster_num = {}
    ir_cluster_num = {}
    lenth_ratio=0
    lam=0.5
    # Main training loop.
    for epoch in range(loop_start_epoch, args.epochs):
        cra_confidence = None

        if (epoch == trainer.cmlabel) and (not resume_full_state): # Enter original Stage-2 at the configured epoch.
            # Load the best checkpoint saved before Stage-2.
            stage1_checkpoint_path = args.stage2_init if stage2_model_only else osp.join(args.logs_dir, '20model_best.pth.tar')
            checkpoint = load_checkpoint(stage1_checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            print('[STAGE2-HANDOFF] loaded model weights for Stage-2 from {}'.format(stage1_checkpoint_path))
            if (not stage2_model_only) and (not getattr(args, 'disable_stage2_handoff_save', False)):
                handoff_path = osp.join(args.logs_dir, args.stage2_handoff_name)
                save_checkpoint(
                    build_full_state_payload(epoch, epoch, source_stage1_checkpoint=stage1_checkpoint_path),
                    False,
                    fpath=handoff_path,
                )
                print('[STAGE2-HANDOFF] saved full-state handoff to {}'.format(handoff_path))
        
        with torch.no_grad():
            ir_eps = float(args.eps)
            rgb_eps = float(args.eps)
            print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
            cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
            print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
            cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(args, dataset_rgb, args.height, args.width,
                                             256, args.workers,
                                             testset=sorted(dataset_rgb.train))
            features_rgb, features_rgb_s = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            
            # Coverage check before torch.cat.
            need = [f for f,_,_ in sorted(dataset_rgb.train)]
            miss_s = [f for f in need if f not in features_rgb_s]
            miss   = [f for f in need if f not in features_rgb]  # Also check features_rgb coverage.

            print(f"[COVER] RGB_s have {len(features_rgb_s)} / need {len(need)} ; miss {len(miss_s)}")
            if miss_s[:10]: print("[COVER] RGB_s missing examples:", miss_s[:10])

            print(f"[COVER] RGB   have {len(features_rgb)} / need {len(need)} ; miss {len(miss)}")
            if miss[:10]: print("[COVER] RGB missing examples:", miss[:10])



            features_rgb_s = torch.cat([features_rgb_s[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0) # Concatenate flipped-view RGB features.

            
            del cluster_loader_rgb
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb_ori=features_rgb
            
            features_rgb_s_=F.normalize(features_rgb_s, dim=1)
            features_rgb_ori_=F.normalize(features_rgb_ori, dim=1)
            # features_rgb_ = torch.cat((features_rgb_,features_rgb_s_), 1)
            features_rgb = torch.cat((features_rgb,features_rgb_s), 1)
            features_rgb_=F.normalize(features_rgb, dim=1)


            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(args, dataset_ir, args.height, args.width,
                                             256, args.workers,
                                             testset=sorted(dataset_ir.train))
            features_ir, features_ir_s = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_ori=features_ir
            features_ir_s = torch.cat([features_ir_s[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_s_=F.normalize(features_ir_s, dim=1)
            features_ir = torch.cat((features_ir,features_ir_s), 1)
            features_ir_=F.normalize(features_ir, dim=1)
            features_ir_ori_=F.normalize(features_ir_ori, dim=1)
            
            
            all_feature = []
            rerank_dist_ir = compute_jaccard_distance(features_ir_, k1=30, k2=args.k2,search_option=3)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            if epoch >= trainer.cmlabel:
                k1_rgb = int(args.stage2_k1)
                iters = 100
            else:
                k1_rgb = int(args.k1)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb_, k1=k1_rgb, k2=args.k2,search_option=3)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)

            del rerank_dist_rgb
            del rerank_dist_ir
            pseudo_labels_all = []
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
        # Generate cluster centers and cluster memories.
        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir_ori)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb_ori)
        memory_ir = ClusterMemory(768, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(768, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb
        del memory_ir, memory_rgb
        wise_momentum=0.9
        print('wise_momentum',wise_momentum)
        # Instance memory stores per-sample features.
        wise_memory_rgb = Memory_wise_v3(768, len(dataset_rgb.train),num_cluster_rgb,temp=args.temp, momentum=wise_momentum).cuda()#args.momentum
        wise_memory_ir = Memory_wise_v3(768, len(dataset_ir.train),num_cluster_ir,temp=args.temp, momentum=wise_momentum).cuda()
        wise_memory_ir.features = F.normalize(features_ir_ori, dim=1).cuda()
        wise_memory_rgb.features = F.normalize(features_rgb_ori, dim=1).cuda()

        nameMap_ir = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_ir.train))}

        nameMap_rgb = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_rgb.train))}

        wise_memory_rgb.labels =  torch.from_numpy(pseudo_labels_rgb)
        wise_memory_ir.labels = torch.from_numpy(pseudo_labels_ir)

        trainer.wise_memory_ir = wise_memory_ir
        trainer.wise_memory_rgb = wise_memory_rgb
        trainer.nameMap_ir=nameMap_ir
        trainer.nameMap_rgb=nameMap_rgb
        del wise_memory_ir, wise_memory_rgb


######## 
        # Generate cluster centers and cluster memories for flipped-view features.
        cluster_features_ir_s = generate_cluster_features(pseudo_labels_ir, features_ir_s)
        cluster_features_rgb_s = generate_cluster_features(pseudo_labels_rgb, features_rgb_s)

        memory_ir_s = ClusterMemory(768, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb_s = ClusterMemory(768, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir_s.features = F.normalize(cluster_features_ir_s, dim=1).cuda()
        memory_rgb_s.features = F.normalize(cluster_features_rgb_s, dim=1).cuda()

        trainer.memory_ir_s = memory_ir_s
        trainer.memory_rgb_s = memory_rgb_s
        del memory_ir_s, memory_rgb_s

        wise_memory_rgb_s = Memory_wise_v3(768, len(dataset_rgb.train),num_cluster_rgb,temp=args.temp, momentum=wise_momentum).cuda()#0.9
        wise_memory_ir_s = Memory_wise_v3(768, len(dataset_ir.train),num_cluster_ir,temp=args.temp, momentum=wise_momentum).cuda()#args.momentum
        wise_memory_ir_s.features = F.normalize(features_ir_s, dim=1).cuda()
        wise_memory_rgb_s.features = F.normalize(features_rgb_s, dim=1).cuda()
        trainer.wise_memory_ir_s = wise_memory_ir_s
        trainer.wise_memory_rgb_s = wise_memory_rgb_s
        del wise_memory_ir_s, wise_memory_rgb_s


        # Split normal samples and outliers according to pseudo labels.
        pseudo_labeled_dataset_ir = []
        ir_label=[]
        pseudo_real_ir = {}
        cams_ir = []
        modality_ir = []
        outlier=0
        cross_cam=[]
        idxs_ir=[]
        ir_cluster=collections.defaultdict(list)

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            cams_ir.append(cid)
            modality_ir.append(1)
            cross_cam.append(int(cid+4))
            ir_label.append(label.item())
            ir_cluster[cid].append(label.item())
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                
                pseudo_real_ir[label.item()] = pseudo_real_ir.get(label.item(),[])+[_]
                pseudo_real_ir[label.item()] = list(set(pseudo_real_ir[label.item()]))

            else:
                outlier=outlier+1


        print('==> Statistics for IR epoch {}: {} clusters outlier {}'.format(epoch, num_cluster_ir,outlier))

        pseudo_labeled_dataset_rgb = []
        rgb_label=[]
        pseudo_real_rgb = {}
        cams_rgb = []
        modality_rgb = []
        outlier=0
        idxs_rgb=[]
        rgb_cluster=collections.defaultdict(list)

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            cams_rgb.append(cid)
            modality_rgb.append(0)
            cross_cam.append(int(cid))
            rgb_label.append(label.item())
            rgb_cluster[cid].append(label.item())
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                
                pseudo_real_rgb[label.item()] = pseudo_real_rgb.get(label.item(),[])+[_]
                pseudo_real_rgb[label.item()] = list(set(pseudo_real_rgb[label.item()]))
            else:
                outlier=outlier+1


        print('==> Statistics for RGB epoch {}: {} clusters outlier {} '.format(epoch, num_cluster_rgb,outlier))
        pseudo_labels_rgb_ori = torch.from_numpy(pseudo_labels_rgb)
      

        if epoch >= trainer.cmlabel:
            with torch.no_grad():
                # Top-k similarity scores and indices used by the soft-structure smoothing step.
                TOPK2 = 20
                Score_TOPK = 20
                cluster_label_ir_self=trainer.wise_memory_ir.labels.detach().cpu() # IR pseudo labels.
                
                ins_sim_rgb_ir = features_rgb_ori_.mm(features_ir_ori_.t()) # Cross-modal similarity.
                topk, ins_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK)) # Top-k values and indices.
                cluster_label_rgb_ir = cluster_label_ir_self[ins_indices_rgb_ir].detach().cpu() # Labels of top-k IR neighbours.
                
                ins_sim_rgb_ir_s = features_rgb_s_.mm(features_ir_s_.t())
                topk, ins_indices_rgb_ir_s = torch.topk(ins_sim_rgb_ir_s, int(Score_TOPK))#20
                ins_label_rgb_ir = cluster_label_ir_self[ins_indices_rgb_ir_s].detach().cpu()
                
                # Equation 25.
                intersect_count_list=[] # Intersection counts.
                for l in range(TOPK2):
                    intersect_count=(ins_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu() # Intersection count for each RGB sample.
                    intersect_count_list.append(intersect_count)

                intersect_count_list = torch.cat(intersect_count_list,1) # Concatenate all intersection counts into a 2D tensor.
                intersect_count, _ = intersect_count_list.max(1)
                cra_confidence = intersect_count.float() / float(TOPK2)
                
                topk,cluster_label_index = torch.topk(intersect_count_list,1) # Index of the largest intersection count.
                
                cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir, dim=1, index=cluster_label_index.view(-1,1)).cpu().numpy()
                cluster_label_rgb_ir= torch.from_numpy(cluster_label_rgb_ir)
                print('soft structure smooth v3')
                rgb_cm_label = cluster_label_rgb_ir.view(-1)+1
                lp_feat_rgb = features_rgb_ori_
                lp_feat_rgb_s = features_rgb_s_

                rgb_cm_label = F.one_hot(rgb_cm_label.view(lp_feat_rgb.size(0),1).long(),int(num_cluster_ir)+1).float().squeeze(1) 

                rgb_self_sim = torch.mm(lp_feat_rgb,lp_feat_rgb.t())
                rgb_self_sim_s = torch.mm(lp_feat_rgb_s,lp_feat_rgb_s.t())

                rgb_self_sim = rgb_self_sim+rgb_self_sim_s

                # Equation 30.
                topk_self, indices_self = torch.topk(rgb_self_sim, 5) 
                mask_self = torch.zeros_like(rgb_self_sim)
                mask_self = mask_self.scatter(1, indices_self, 1)
                rgb_self_sim    = mask_self

                # After smoothing.
                smooth_rgb = torch.mm(rgb_self_sim.cpu(),rgb_cm_label.cpu()) 
                smooth_rgb = torch.argmax(smooth_rgb,1).view(-1).numpy()    
                pseudo_labels_rgb_cm = [int(smolabel-1) for smolabel in smooth_rgb]
                pseudo_labels_rgb_cm = np.array(pseudo_labels_rgb_cm)
                cluster_label_rgb_ir= torch.from_numpy(pseudo_labels_rgb_cm)

                if getattr(args, 'enable_cglf', False):
                    # Optional CGLF: disabled by default in metric-first mode.
                    cglf_threshold = float(getattr(args, 'cglf_threshold', 0.0))
                    low_conf_mask = (cra_confidence < cglf_threshold)
                    n_filtered = int(low_conf_mask.sum().item())
                    n_total = len(cra_confidence)
                    cluster_label_rgb_ir_np = cluster_label_rgb_ir.view(-1).cpu().numpy().copy()
                    cluster_label_rgb_ir_np[low_conf_mask.numpy()] = -1
                    cluster_label_rgb_ir = torch.from_numpy(cluster_label_rgb_ir_np)
                    print('[CGLF] epoch={} threshold={:.2f} conf(mean/p50/p90)={:.3f}/{:.3f}/{:.3f} filtered={}/{} ({:.1%})'.format(
                        epoch, cglf_threshold,
                        float(cra_confidence.mean().item()),
                        float(torch.quantile(cra_confidence, 0.50).item()),
                        float(torch.quantile(cra_confidence, 0.90).item()),
                        n_filtered, n_total, n_filtered / max(n_total, 1)
                    ))

                if getattr(args, 'enable_stage2_tailtrim', False):
                    tailtrim_start = trainer.cmlabel + int(getattr(args, 'stage2_tailtrim_delay', 3))
                    tailtrim_pct = float(getattr(args, 'stage2_tailtrim_pct', 0.02))
                    tailtrim_warmup = max(1, int(getattr(args, 'stage2_tailtrim_warmup', 4)))
                    if epoch >= tailtrim_start:
                        ramp = min(1.0, float(epoch - tailtrim_start + 1) / float(tailtrim_warmup))
                    else:
                        ramp = 0.0
                    effective_tailtrim_pct = tailtrim_pct * ramp
                    tailtrim_decay_after = int(getattr(args, 'stage2_tailtrim_decay_after', -1))
                    tailtrim_decay_to = float(getattr(args, 'stage2_tailtrim_decay_to', tailtrim_pct))
                    tailtrim_decay_warmup = max(1, int(getattr(args, 'stage2_tailtrim_decay_warmup', 4)))
                    if tailtrim_decay_after >= 0 and epoch > tailtrim_decay_after:
                        decay_ramp = min(
                            1.0,
                            float(epoch - tailtrim_decay_after) / float(tailtrim_decay_warmup)
                        )
                        decay_to = min(tailtrim_pct, max(0.0, tailtrim_decay_to))
                        effective_tailtrim_pct = (
                            effective_tailtrim_pct * (1.0 - decay_ramp)
                            + decay_to * decay_ramp
                        )
                    n_total = len(cra_confidence)
                    n_trim = int(n_total * effective_tailtrim_pct)
                    if (epoch >= tailtrim_start) and (n_trim > 0):
                        tail_indices = torch.topk(-cra_confidence.float(), k=n_trim, dim=0).indices.view(-1)
                        tail_mask = torch.zeros_like(cra_confidence, dtype=torch.bool)
                        tail_mask[tail_indices] = True
                        cluster_label_rgb_ir_np = cluster_label_rgb_ir.view(-1).cpu().numpy().copy()
                        cluster_label_rgb_ir_np[tail_mask.numpy()] = -1
                        cluster_label_rgb_ir = torch.from_numpy(cluster_label_rgb_ir_np)
                        print('[TAIL] epoch={} start={} warmup={} decay_after={} decay_to={:.3f} decay_warmup={} pct={:.3f} effective_pct={:.4f} filtered={}/{} conf(mean/p10/p50)={:.3f}/{:.3f}/{:.3f}'.format(
                            epoch, int(tailtrim_start), int(tailtrim_warmup),
                            int(tailtrim_decay_after), tailtrim_decay_to, int(tailtrim_decay_warmup),
                            tailtrim_pct, effective_tailtrim_pct, int(tail_mask.sum().item()), n_total,
                            float(cra_confidence.mean().item()),
                            float(torch.quantile(cra_confidence, 0.10).item()),
                            float(torch.quantile(cra_confidence, 0.50).item()),
                        ))

                del rgb_self_sim,smooth_rgb,lp_feat_rgb,lp_feat_rgb_s


            # Number of unique pseudo-label clusters.
            lamda_cm=0.1
            pseudo_labels_rgb=cluster_label_rgb_ir.view(-1).cpu().numpy() # Convert to a one-dimensional NumPy array.
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)

            # Rebuild the pseudo-labeled datasets.
            pseudo_labeled_dataset_ir = []
            # ir_label=[]
            # pseudo_real_ir = {}
            cams_ir = []
            modality_ir = []
            cross_cam=[]
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
                cams_ir.append(int(cid+4))
                modality_ir.append(int(1))
                cross_cam.append(int(cid+4))
                indexes = torch.tensor([trainer.nameMap_ir[fname]])
                ir_label_ms = trainer.wise_memory_ir.labels[indexes]

                if (label != -1) and (ir_label_ms!= -1):
                    pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                    # if epoch%10 == 0:
                    #     print(fname,label.item())
            print('stage2 ==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

            pseudo_labeled_dataset_rgb = []
            # rgb_label=[]
            # pseudo_real_rgb = {}
            cams_rgb = []
            modality_rgb = []
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
                cams_rgb.append(int(cid))
                modality_rgb.append(int(0))
                cross_cam.append(int(cid))
                indexes = torch.tensor([trainer.nameMap_rgb[fname]])
                rgb_label_ms = trainer.wise_memory_rgb.labels[indexes] # 

                if (label != -1) and (rgb_label_ms!= -1):
                    pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                    # if epoch%10 == 0:
                    #     print(fname,label.item())
            print('stage2 ==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))


            features_all = torch.cat((features_rgb_ori,features_ir_ori),dim=0)

            pseudo_labels_all = torch.cat((torch.from_numpy(pseudo_labels_rgb),torch.from_numpy(pseudo_labels_ir)),dim=-1).view(-1).cpu().numpy()

            cluster_features_ir = generate_cluster_features(pseudo_labels_all, features_all)

############## Shared memory layer
            shared_memory = ClusterMemory(768, num_cluster_ir, temp=args.temp,
                                   momentum=0.1, use_hard=args.use_hard)#.cuda()
            shared_memory.features = F.normalize(cluster_features_ir, dim=1).cuda()


            trainer.memory_ir = shared_memory
            trainer.memory_rgb = shared_memory
            features_all_s = torch.cat((features_rgb_s,features_ir_s),dim=0)
            cluster_features_ir_s = generate_cluster_features(pseudo_labels_all, features_all_s)
            shared_memory_s = ClusterMemory(768 , num_cluster_ir, temp=args.temp,
                                   momentum=0.1, use_hard=args.use_hard)

            shared_memory_s.features = F.normalize(cluster_features_ir_s, dim=1).cuda()

            trainer.memory_rgb_s = shared_memory_s
            trainer.memory_ir_s = shared_memory_s
            del shared_memory, shared_memory_s

        trainer.sample_weight_ir = None
        trainer.sample_weight_rgb = None

        if getattr(args, 'enable_stage2_softweight', False) and (epoch >= trainer.cmlabel) and (cra_confidence is not None):
            rgb_soft_weight = build_stage2_rgb_soft_weight(
                cra_confidence.cpu(),
                min_weight=float(getattr(args, 'stage2_softweight_min', 0.80)),
                power=float(getattr(args, 'stage2_softweight_power', 1.0))
            )
            trainer.sample_weight_rgb = rgb_soft_weight.cuda()
            print('[S2W] epoch={} min={:.3f} power={:.3f} RGB(mean/min/max)={:.3f}/{:.3f}/{:.3f}'.format(
                epoch,
                float(getattr(args, 'stage2_softweight_min', 0.80)),
                float(getattr(args, 'stage2_softweight_power', 1.0)),
                float(rgb_soft_weight.mean().item()),
                float(rgb_soft_weight.min().item()),
                float(rgb_soft_weight.max().item()),
            ))
            del rgb_soft_weight

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                    ir_batch, args.workers, args.num_instances, iters,
                                    trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)
        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                rgb_batch, args.workers, args.num_instances, iters,
                                trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)


        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()
        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer, print_freq=args.print_freq, train_iters=len(train_loader_ir))

        if epoch>=0 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            _,mAP_homo = evaluator.evaluate(test_loader_ir, dataset_ir.query, dataset_ir.gallery, cmc_flag=True,modal=2)
            _,mAP_homo = evaluator.evaluate(test_loader_rgb, dataset_rgb.query, dataset_rgb.gallery, cmc_flag=True,modal=1)
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode='all'
            print(mode)
            data_path='/home/lhp/project/DATASETS/SYSU-MM01'
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
            nquery = len(query_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            for trial in range(10):
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
                ngall = len(gall_label)
                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

                gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
                ######match

                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP

                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

                print('Test Trial: {}'.format(trial))
                print(
                    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            cmc = all_cmc / 10
            mAP = all_mAP / 10
            mINP = all_mINP / 10
            print('All Average:')
            print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

#################################
            # delay checkpoint selection until we also have indoor mAP/mINP
            allsearch_cmc = cmc
            allsearch_mAP = mAP
            allsearch_mINP = mINP


            mode='indoor'
            # data_path='/home/kangpeipei/lhp/SDCL/data/SYSU-MM01'
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
            nquery = len(query_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            for trial in range(10):
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
                ngall = len(gall_label)
                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

                gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP

                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP


                print('Test Trial: {}'.format(trial))
                print(
                    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            cmc = all_cmc / 10
            mAP = all_mAP / 10
            mINP = all_mINP / 10
            print('indoor All Average:')
            print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            indoor_cmc = cmc
            indoor_mAP = mAP
            indoor_mINP = mINP

#################################
            # Select model_best by mAP/mINP instead of Rank-1.
            # best_select_mode:
            # - legacy (default): matches earlier scripts for comparable best_score values.
            #   score = 0.60*all_mAP + 0.40*indoor_mAP + 0.10*all_mINP
            # - full: also includes indoor_mINP and breaks ties with indoor_mAP + indoor_mINP.
            tie = float(indoor_mAP) + float(indoor_mINP)
            sel = getattr(args, 'best_select_mode', 'legacy')

            def score_for_mode(mode):
                if mode == 'full':
                    return (
                        0.42 * float(allsearch_mAP)
                        + 0.28 * float(indoor_mAP)
                        + 0.12 * float(allsearch_mINP)
                        + 0.18 * float(indoor_mINP)
                    )
                return (
                    0.60 * float(allsearch_mAP)
                    + 0.40 * float(indoor_mAP)
                    + 0.10 * float(allsearch_mINP)
                )

            score = score_for_mode(sel)
            is_best = (score > best_score + 1e-8) or (
                abs(score - best_score) <= 1e-8 and tie > best_tiebreak + 1e-8
            )
            if is_best:
                best_score = float(score)
                best_tiebreak = tie
                best_epoch = int(epoch)

            # stage1 (epoch < cmlabel) uses an independent selector for 20model_best.
            stage1_mode_raw = getattr(args, 'stage1_best_select_mode', 'legacy')
            stage1_sel = sel if stage1_mode_raw == 'follow' else stage1_mode_raw
            stage1_score = score_for_mode(stage1_sel)
            stage1_is_best = False
            if epoch < trainer.cmlabel:
                stage1_is_best = (stage1_score > best_stage1_score + 1e-8) or (
                    abs(stage1_score - best_stage1_score) <= 1e-8 and tie > best_stage1_tiebreak + 1e-8
                )
                if stage1_is_best:
                    best_stage1_score = float(stage1_score)
                    best_stage1_tiebreak = tie
                    best_stage1_epoch = int(epoch)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_score': float(score),
                'allsearch_mAP': float(allsearch_mAP),
                'allsearch_mINP': float(allsearch_mINP),
                'indoor_mAP': float(indoor_mAP),
                'indoor_mINP': float(indoor_mINP),
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            if epoch < trainer.cmlabel:
                save_checkpoint10({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_score': float(stage1_score),
                }, stage1_is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            # is_best = (mAP > best_mAP)
            # best_mAP = max(mAP, best_mAP)
            # save_checkpoint({
            #     'state_dict': model.state_dict(),
            #     'epoch': epoch + 1,
            #     'best_mAP': best_mAP,
            # }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            # save_checkpoint_match(matcher_rgb, is_best, fpath=osp.join(args.logs_dir, 'matcher_rgb_checkpoint.pkl'),match='rgb')

            # save_checkpoint_match(matcher_ir, is_best, fpath=osp.join(args.logs_dir, 'matcher_ir_checkpoint.pkl'),match='ir')
            # save_checkpoint_match({
            #     'state_dict': matcher_rgb.state_dict(),
            #     'epoch': epoch + 1,
            #     'best_mAP': best_mAP,
            # }, is_best, fpath=osp.join(args.logs_dir, 'matcher_rgb_checkpoint.pth.tar'),match='rgb')

            # save_checkpoint_match({
            #     'state_dict': matcher_ir.state_dict(),
            #     'epoch': epoch + 1,
            #     'best_mAP': best_mAP,
            # }, is_best, fpath=osp.join(args.logs_dir, 'matcher_ir_checkpoint.pth.tar'),match='ir')


            print('\n * Finished epoch {:3d}  sel={}  all_mAP: {:5.1%} indoor_mAP: {:5.1%} '
                  'indoor_mINP: {:5.1%} score: {:.4f}  best@epoch={} best_score: {:.4f}{}\n'.format(
                      epoch, getattr(args, 'best_select_mode', 'legacy'),
                      allsearch_mAP, indoor_mAP, indoor_mINP, score,
                      best_epoch if best_epoch >= 0 else '-', best_score, ' *' if is_best else ''))
############################
        lr_scheduler.step()
        # if epoch >30:
        #     break
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    be = checkpoint.get('epoch', '?')
    bs = checkpoint.get('best_score', float('nan'))
    print('=> model_best from epoch {} (train loop epoch index {}), checkpoint best_score={:.4f}, '
          'stored all_mAP={:.2%} indoor_mAP={:.2%} indoor_mINP={:.2%}'.format(
              be, int(be) - 1 if isinstance(be, int) else be, float(bs),
              float(checkpoint.get('allsearch_mAP', 0)),
              float(checkpoint.get('indoor_mAP', 0)),
              float(checkpoint.get('indoor_mINP', 0))))
    model.load_state_dict(checkpoint['state_dict'])
    _,mAP_homo = evaluator.evaluate(test_loader_ir, dataset_ir.query, dataset_ir.gallery, cmc_flag=True,modal=2)
    _,mAP_homo = evaluator.evaluate(test_loader_rgb, dataset_rgb.query, dataset_rgb.gallery, cmc_flag=True,modal=1)
    mode='all'
    # data_path='/home/kangpeipei/lhp/SDCL/data/SYSU-MM01'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('all search All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))


    mode='indoor'
    # data_path='/home/kangpeipei/lhp/SDCL/data/SYSU-MM01'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('indoor All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

#################################
    # is_best = (mAP > best_mAP)
    # best_mAP = max(mAP, best_mAP)
    # save_checkpoint({
    #     'state_dict': model.state_dict(),
    #     'epoch': epoch + 1,
    #     'best_mAP': best_mAP,
    # }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

    # print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
    #       format(epoch, mAP, best_mAP, ' *' if is_best else ''))
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    parser.add_argument(
        "--config_file", default="vit_base_ics_288.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2,
                        help='SYSU train/test DataLoader batch size; both IR and RGB training loaders use this value.')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='If greater than 1, divide loss by N and call optimizer.step every N iterations; memory remains controlled by -b; '
                             'the number of optimizer steps per epoch becomes iters/N, so increase --epochs proportionally if needed.')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=288, help="input height")#288 384
    parser.add_argument('--width', type=int, default=144, help="input width")#144 128
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,#30
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--stage2-k1', type=int, default=15,
                        help='RGB clustering jaccard k1 used when epoch >= cmlabel; original source uses 15, alternatives such as 12/18 can be tested.')

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        )
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument(
        '--logs-dir', type=str, metavar='PATH', default=None,
        help='Log and checkpoint directory; default is ./logs/sysu_train generated by main_worker from log_name. '
             'Grid search should pass this explicitly, for example --logs-dir ./logs/sysu_grid/ls0.15_rc0.05_alt',
    )
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    parser.add_argument('--warmup-step', type=int, default=0)
    parser.add_argument('--milestones', nargs='+', type=int, default=[20,40],
                        help='milestones for the learning rate decay')
    parser.add_argument(
        '--cross-modal-mode',
        type=str,
        default='alternating',
        choices=('rgb2ir', 'alternating', 'both'),
        help='Cross-modal neighbourhood mode: alternating matches the source epoch%%2 schedule; rgb2ir uses only RGB-to-IR each step; both uses both directions for ablation.',
    )
    parser.add_argument(
        '--trainer-backend',
        type=str,
        default='source',
        choices=('source',),
        help='Training backbone: fixed source trainer for metric-first stability.',
    )
    parser.add_argument(
        '--best-select-mode',
        type=str,
        default='full',
        choices=('legacy', 'full'),
        help='Select model_best: full includes indoor_mINP and tie-breaks by indoor_mAP+mINP; legacy uses 0.6*all+0.4*indoor+0.1*all_mINP and also tie-breaks by indoor metrics.',
    )
    parser.add_argument(
        '--stage1-best-select-mode',
        type=str,
        default='legacy',
        choices=('legacy', 'full', 'follow'),
        help='Select 20model_best before cmlabel for Stage-2 warm-start: legacy is the stable default; full matches final selection; follow mirrors --best-select-mode.',
    )
    parser.add_argument('--cmlabel', type=int, default=30,
                        help='stage2 start epoch')
    parser.add_argument('--stage2-resume-full-state', type=str, default='',
                        help='Resume Stage-2 from a v35 full-state handoff checkpoint, including optimizer, scheduler, RNG, and best-score state.')
    parser.add_argument('--stage2-handoff-name', type=str, default='stage2_handoff_full_state.pth.tar',
                        help='Filename written in --logs-dir at the Stage-1 -> Stage-2 boundary for exact full-state resume.')
    parser.add_argument('--disable-stage2-handoff-save', action='store_true',
                        help='Do not write the full-state Stage-2 handoff checkpoint during a full-chain run.')
    parser.add_argument('--no-resume-rng', action='store_true',
                        help='When using --stage2-resume-full-state, skip restoring Python/NumPy/Torch RNG states.')
    parser.add_argument('--stage2-only', action='store_true',
                        help='Compatibility mode: start at cmlabel from a model-only 20model_best checkpoint. Prefer full-chain or --stage2-resume-full-state for v35.')
    parser.add_argument('--stage2-init', type=str, default='',
                        help='Model-only checkpoint used with --stage2-only.')
    parser.add_argument('--enable-cglf', action='store_true',
                        help='Explicitly enable CGLF; metric-first defaults keep it disabled.')
    parser.add_argument('--cglf-threshold', type=float, default=0.0,
                        help='CGLF: CRA confidence threshold (0-1). Samples below this are excluded from stage2 training.')
    parser.add_argument('--enable-stage2-softweight', action='store_true',
                        help='Enable CRA-guided soft weighting in stage2 RGB memory learning.')
    parser.add_argument('--stage2-softweight-min', dest='stage2_softweight_min', type=float, default=0.80,
                        help='Minimum RGB sample weight in stage2 soft weighting.')
    parser.add_argument('--stage2-softweight-power', dest='stage2_softweight_power', type=float, default=1.0,
                        help='Confidence sharpening power for stage2 soft weighting.')
    parser.add_argument('--enable-stage2-tailtrim', action='store_true',
                        help='Drop the lowest-confidence RGB tail in stage2 before soft weighting.')
    parser.add_argument('--stage2-tailtrim-delay', dest='stage2_tailtrim_delay', type=int, default=3,
                        help='Delay epochs after cmlabel before tail trimming starts.')
    parser.add_argument('--stage2-tailtrim-warmup', dest='stage2_tailtrim_warmup', type=int, default=4,
                        help='Epochs used to linearly ramp tail trimming from 0 to --stage2-tailtrim-pct.')
    parser.add_argument('--stage2-tailtrim-pct', dest='stage2_tailtrim_pct', type=float, default=0.02,
                        help='Target fraction of lowest-confidence RGB samples to remove in stage2 after warmup.')
    parser.add_argument('--stage2-tailtrim-decay-after', dest='stage2_tailtrim_decay_after', type=int, default=50,
                        help='After this epoch index, linearly decay effective tailtrim pct toward --stage2-tailtrim-decay-to.')
    parser.add_argument('--stage2-tailtrim-decay-to', dest='stage2_tailtrim_decay_to', type=float, default=0.01,
                        help='Target effective tailtrim pct after late-stage decay.')
    parser.add_argument('--stage2-tailtrim-decay-warmup', dest='stage2_tailtrim_decay_warmup', type=int, default=4,
                        help='Epochs used to linearly decay tailtrim pct after --stage2-tailtrim-decay-after.')
    main()
