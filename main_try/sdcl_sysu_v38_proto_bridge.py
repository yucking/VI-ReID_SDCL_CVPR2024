# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
"""
SDCL v38: Stage-1 Cross-Modal Prototype Bridge.

Stage-1 keeps the original per-modality DBSCAN/memory loop, then adds a
conservative bridge memory only for mutual nearest RGB/IR cluster prototypes.
Unmatched or low-confidence clusters keep the original SDCL objective only.
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
best_tiebreak = -1.0  # 同分用 indoor_mAP + indoor_mINP 打破平局
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
    这个函数用于获取数据集。它接收数据集的名称（name）和数据存储的根目录（data_dir），然后使用datasets.create方法创建数据集对象并返回。
    '''
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):
    '''
    这个函数用于创建红外训练数据加载器。
    train_set：排序后的训练数据集。
    rmgs_flag：是否使用多画廊采样器。
    sampler：根据no_cam标志选择相应的采样器。
    train_loader：使用DataLoader和IterLoader创建训练数据加载器。
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
    这个函数用于创建彩色训练数据加载器。它与get_train_loader_ir类似，但支持双重变换（train_transformer和train_transformer1）
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
    这个函数用于创建测试数据加载器。它使用标准化变换（normalizer）对图像进行预处理，并使用DataLoader加载数据。
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
    这个函数用于创建模型。它接收一些参数（如架构类型、特征数量、dropout率等），并使用models.create方法创建模型。创建的模型会被转移到GPU上并进行数据并行化处理。
    '''
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)
    return model


class TestData(data.Dataset):
    '''
    这个类定义了一个测试数据集。它会将给定的测试图像文件列表和标签列表进行预处理，并定义了获取单个数据项和数据集长度的方法。
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
    这个函数用于处理SYSU-MM01数据集的查询集。它会根据给定的模式（all或indoor）选择红外摄像机，并读取图像文件路径和标签。
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
    这个函数用于处理SYSU-MM01数据集的图库。它会根据给定的模式（all或indoor）选择RGB摄像机，并读取图像文件路径和标签。
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
    '''flip horizontal:这个函数用于水平翻转图像。它通过索引选择的方式将图像在宽度方向上进行翻转'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
def extract_gall_feat(model,gall_loader,ngall):
    '''
    这个函数用于提取图库特征。它将模型设置为评估模式（eval），然后从数据加载器中批量读取数据，通过模型前向传播得到特征，并进行翻转操作和归一化处理，最终将特征保存到数组中。
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
    
    l2norm = Normalize(2) # 创建了一个 L2 归一化层，用于后续处理数据时的归一化操作
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

    # 仅当未传 --logs-dir 时使用 ./logs/<log_name>（默认 sysu_train）。
    # 网格脚本传入 --logs-dir 时不得再覆盖，否则会全部写到 sysu_train 并互相覆盖。
    if args.logs_dir is None:
        args.logs_dir = osp.abspath(osp.join('./logs', log_name))
    else:
        args.logs_dir = osp.abspath(args.logs_dir)
    print('==> logs_dir (final): {}'.format(args.logs_dir))
    start_time = time.monotonic()

    cudnn.deterministic = False
    cudnn.benchmark = True
    print('==> memsafe_mode: cudnn benchmark enabled; dataloader seeds fixed')
    
    # 设置日志记录器，将输出信息保存到日志文件中
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
    # 将模型移动到GPU上，并使用 nn.DataParallel 包装模型以支持多GPU训练。
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
    # 创建模型参数的列表，只包含那些需要梯度的参数
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]

    # 创建一个随机梯度下降（SGD）优化器和一个学习率调度器，用于在训练过程中调整学习率
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # 创建一个评估器对象，用于评估模型的性能
    evaluator = Evaluator(model)

    @torch.no_grad()
    def generate_cluster_features(labels, features):
        # 创建一个名为 centers 的字典，用于存储不同标签的中心特征
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels): # 遍历标签和特征，如果标签不是 -1（表示未标记或异常值），则将特征添加到对应的标签列表中
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())  # 对每个标签的特征列表进行堆叠和求平均，得到该标签的中心特征
        ]
        # 将所有中心特征堆叠成一个张量，并返回
        centers = torch.stack(centers, dim=0)
        return centers

    @torch.no_grad()
    def build_stage1_proto_bridge(pseudo_labels_rgb, pseudo_labels_ir,
                                  cluster_features_rgb, cluster_features_ir,
                                  cluster_features_rgb_s, cluster_features_ir_s,
                                  min_sim=0.42, min_margin=0.015, min_cluster_size=4,
                                  max_pairs=512):
        rgb_labels = torch.as_tensor(pseudo_labels_rgb, dtype=torch.long)
        ir_labels = torch.as_tensor(pseudo_labels_ir, dtype=torch.long)
        bridge_rgb = torch.full_like(rgb_labels, -1)
        bridge_ir = torch.full_like(ir_labels, -1)
        stats = {
            'pairs': 0,
            'rgb_coverage': 0.0,
            'ir_coverage': 0.0,
            'rgb_cluster_coverage': 0.0,
            'ir_cluster_coverage': 0.0,
            'mean_sim': 0.0,
            'min_sim': 0.0,
            'mean_margin': 0.0,
        }
        if cluster_features_rgb.numel() == 0 or cluster_features_ir.numel() == 0:
            return bridge_rgb, bridge_ir, None, None, stats

        rgb_proto = F.normalize(cluster_features_rgb.float(), dim=1)
        ir_proto = F.normalize(cluster_features_ir.float(), dim=1)
        sim = rgb_proto.mm(ir_proto.t())
        num_rgb, num_ir = sim.size()
        rgb_valid_labels = rgb_labels[rgb_labels >= 0]
        ir_valid_labels = ir_labels[ir_labels >= 0]
        rgb_counts = torch.bincount(rgb_valid_labels, minlength=num_rgb)[:num_rgb]
        ir_counts = torch.bincount(ir_valid_labels, minlength=num_ir)[:num_ir]

        rgb_top2 = sim.topk(min(2, num_ir), dim=1).values
        ir_top2 = sim.topk(min(2, num_rgb), dim=0).values
        rgb_best_sim, rgb_best_ir = sim.max(dim=1)
        _, ir_best_rgb = sim.max(dim=0)
        rgb_margin = rgb_top2[:, 0] - (rgb_top2[:, 1] if rgb_top2.size(1) > 1 else -1.0)
        ir_margin = ir_top2[0, :] - (ir_top2[1, :] if ir_top2.size(0) > 1 else -1.0)

        candidates = []
        for rgb_idx in range(num_rgb):
            ir_idx = int(rgb_best_ir[rgb_idx].item())
            if int(ir_best_rgb[ir_idx].item()) != rgb_idx:
                continue
            if int(rgb_counts[rgb_idx].item()) < int(min_cluster_size):
                continue
            if int(ir_counts[ir_idx].item()) < int(min_cluster_size):
                continue
            pair_sim = float(rgb_best_sim[rgb_idx].item())
            pair_margin = min(float(rgb_margin[rgb_idx].item()), float(ir_margin[ir_idx].item()))
            if pair_sim < float(min_sim) or pair_margin < float(min_margin):
                continue
            candidates.append((pair_sim, pair_margin, rgb_idx, ir_idx))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if max_pairs > 0:
            candidates = candidates[:int(max_pairs)]
        if not candidates:
            return bridge_rgb, bridge_ir, None, None, stats

        bridge_features = []
        bridge_features_s = []
        sims = []
        margins = []
        for bridge_id, (pair_sim, pair_margin, rgb_idx, ir_idx) in enumerate(candidates):
            bridge_rgb[rgb_labels == int(rgb_idx)] = int(bridge_id)
            bridge_ir[ir_labels == int(ir_idx)] = int(bridge_id)
            bridge_features.append((rgb_proto[rgb_idx] + ir_proto[ir_idx]) * 0.5)
            bridge_features_s.append((
                F.normalize(cluster_features_rgb_s[rgb_idx].float(), dim=0)
                + F.normalize(cluster_features_ir_s[ir_idx].float(), dim=0)
            ) * 0.5)
            sims.append(pair_sim)
            margins.append(pair_margin)

        bridge_features = F.normalize(torch.stack(bridge_features, dim=0), dim=1)
        bridge_features_s = F.normalize(torch.stack(bridge_features_s, dim=0), dim=1)
        stats.update({
            'pairs': len(candidates),
            'rgb_coverage': float((bridge_rgb >= 0).sum().item()) / float(max(1, bridge_rgb.numel())),
            'ir_coverage': float((bridge_ir >= 0).sum().item()) / float(max(1, bridge_ir.numel())),
            'rgb_cluster_coverage': float(len(candidates)) / float(max(1, num_rgb)),
            'ir_cluster_coverage': float(len(candidates)) / float(max(1, num_ir)),
            'mean_sim': float(np.mean(sims)),
            'min_sim': float(np.min(sims)),
            'mean_margin': float(np.mean(margins)),
        })
        return bridge_rgb, bridge_ir, bridge_features, bridge_features_s, stats

    @torch.no_grad()
    def build_stage2_rgb_soft_weight(confidence, min_weight=0.80, power=1.0):
        conf = confidence.float().clamp(0.0, 1.0)
        power = max(float(power), 1e-6)
        conf = conf.pow(power)
        weights = float(min_weight) + (1.0 - float(min_weight)) * conf
        return weights.clamp(float(min_weight), 1.0)

    @torch.no_grad()
    def build_reciprocal_ambiguity_mask(rgb_features, rgb_labels, rgb_features_s,
                                         ir_features, ir_labels, ir_features_s,
                                         num_clusters, topk=2, margin=0.03):
        def modality_centers(features, labels):
            labels = torch.as_tensor(labels, dtype=torch.long)
            valid = (labels >= 0) & (labels < num_clusters)
            centers = torch.zeros(num_clusters, features.size(1), dtype=torch.float32)
            counts = torch.zeros(num_clusters, 1, dtype=torch.float32)
            centers.index_add_(0, labels[valid], features[valid].float())
            counts.index_add_(0, labels[valid], torch.ones(int(valid.sum().item()), 1))
            return F.normalize(centers / counts.clamp_min(1.0), dim=1), counts.view(-1) > 0

        rgb, rgb_valid = modality_centers(rgb_features, rgb_labels)
        ir, ir_valid = modality_centers(ir_features, ir_labels)
        rgb_s, _ = modality_centers(rgb_features_s, rgb_labels)
        ir_s, _ = modality_centers(ir_features_s, ir_labels)
        valid_pairs = rgb_valid[:, None] & ir_valid[None, :]
        k = min(max(1, int(topk)), int(num_clusters))

        def reciprocal_pairs(left, right):
            similarity = left.mm(right.t()).masked_fill(~valid_pairs, float('-inf'))
            left_topk = torch.zeros_like(valid_pairs)
            right_topk = torch.zeros_like(valid_pairs)
            left_topk.scatter_(1, similarity.topk(k, dim=1).indices, True)
            right_topk.scatter_(0, similarity.topk(k, dim=0).indices, True)
            reciprocal = left_topk & right_topk & valid_pairs
            close_to_current = similarity >= (similarity.diag().view(-1, 1) - float(margin))
            return reciprocal & close_to_current, similarity

        pairs, similarity = reciprocal_pairs(rgb, ir)
        pairs_s, _ = reciprocal_pairs(rgb_s, ir_s)
        mask = pairs & pairs_s
        mask.fill_diagonal_(False)
        mask = mask | mask.t()
        valid_count = int((rgb_valid & ir_valid).sum().item())
        edge_count = int(mask.triu(diagonal=1).sum().item())
        density = edge_count / max(valid_count * (valid_count - 1) / 2, 1)
        return mask, edge_count, valid_count, density, float(similarity[valid_pairs].mean().item())

    # 创建了一个颜色抖动的数据增强操作，用于在训练过程中随机变化图像的亮度、对比度、饱和度和色调。
    color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    # 定义了一个标准化层，用于将图像的每个通道减去均值并除以标准差，以实现标准化
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
    # 开始训练循环
    for epoch in range(args.epochs):
        cra_confidence = None

        if (epoch == trainer.cmlabel) :# 30 在特定轮次执行特定操作
            # 加载之前保存的最佳模型检查点
            checkpoint = load_checkpoint(osp.join(args.logs_dir,'20model_best.pth.tar'))
            model.load_state_dict(checkpoint['state_dict'])
        
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
            
            # —— 覆盖率检查（放在 torch.cat 前一行）
            need = [f for f,_,_ in sorted(dataset_rgb.train)]
            miss_s = [f for f in need if f not in features_rgb_s]
            miss   = [f for f in need if f not in features_rgb]  # 如果也要拼 features_rgb

            print(f"[COVER] RGB_s have {len(features_rgb_s)} / need {len(need)} ; miss {len(miss_s)}")
            if miss_s[:10]: print("[COVER] RGB_s missing examples:", miss_s[:10])

            print(f"[COVER] RGB   have {len(features_rgb)} / need {len(need)} ; miss {len(miss)}")
            if miss[:10]: print("[COVER] RGB missing examples:", miss[:10])



            features_rgb_s = torch.cat([features_rgb_s[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0) # 翻转图像特征拼接

            
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
        # 生成聚类特征和聚类内存
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
        # 内存管理类，存储实例
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
        # 生成聚类特征和聚类内存(翻转)
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


        # 根据训练集及其相应的伪标签，分理处正常与异常标签集
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
                # 分别表示要计算的前20个最大值和前20个最大值的索引 （公式23,24）
                TOPK2 = 20
                Score_TOPK = 20
                cluster_label_ir_self=trainer.wise_memory_ir.labels.detach().cpu() # 获取标签
                
                ins_sim_rgb_ir = features_rgb_ori_.mm(features_ir_ori_.t()) # 计算相似性
                topk, ins_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK)) # 获取张量中最大的K个元素及其索引
                cluster_label_rgb_ir = cluster_label_ir_self[ins_indices_rgb_ir].detach().cpu() # 获取对应标签
                
                ins_sim_rgb_ir_s = features_rgb_s_.mm(features_ir_s_.t())
                topk, ins_indices_rgb_ir_s = torch.topk(ins_sim_rgb_ir_s, int(Score_TOPK))#20
                ins_label_rgb_ir = cluster_label_ir_self[ins_indices_rgb_ir_s].detach().cpu()
                
                # （公式25）
                intersect_count_list=[] # 记录交集数量
                for l in range(TOPK2):
                    intersect_count=(ins_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu() # 计算每个样本的交集数量
                    intersect_count_list.append(intersect_count)

                intersect_count_list = torch.cat(intersect_count_list,1) # 将所有交集数量拼接在一起：二维张量
                intersect_count, _ = intersect_count_list.max(1)
                cra_confidence = intersect_count.float() / float(TOPK2)
                
                topk,cluster_label_index = torch.topk(intersect_count_list,1) # 获取张量中最大的元素及其索引
                
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

                # 公式30
                topk_self, indices_self = torch.topk(rgb_self_sim, 5) 
                mask_self = torch.zeros_like(rgb_self_sim)
                mask_self = mask_self.scatter(1, indices_self, 1)
                rgb_self_sim    = mask_self

                # 平滑之后 （26？
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


            # 不同聚类标签的数量
            lamda_cm=0.1
            pseudo_labels_rgb=cluster_label_rgb_ir.view(-1).cpu().numpy() # 转换为一个一维的NumPy数组
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)

            # 生成新的标签集
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

############## 共享层
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
        trainer.enable_proto_bridge = False
        trainer.proto_bridge_weight = 0.0
        trainer.proto_bridge_label_ir = None
        trainer.proto_bridge_label_rgb = None
        trainer.proto_bridge_memory = None
        trainer.proto_bridge_memory_s = None

        if getattr(args, 'enable_proto_bridge', False) and epoch < trainer.cmlabel:
            bridge_label_rgb, bridge_label_ir, bridge_features, bridge_features_s, bridge_stats = build_stage1_proto_bridge(
                pseudo_labels_rgb, pseudo_labels_ir,
                cluster_features_rgb, cluster_features_ir,
                cluster_features_rgb_s, cluster_features_ir_s,
                min_sim=float(getattr(args, 'proto_bridge_min_sim', 0.42)),
                min_margin=float(getattr(args, 'proto_bridge_min_margin', 0.015)),
                min_cluster_size=int(getattr(args, 'proto_bridge_min_cluster_size', 4)),
                max_pairs=int(getattr(args, 'proto_bridge_max_pairs', 512)),
            )
            if bridge_features is not None and int(bridge_stats.get('pairs', 0)) > 0:
                proto_bridge_memory = ClusterMemory(
                    768, bridge_features.size(0), temp=float(getattr(args, 'proto_bridge_temp', args.temp)),
                    momentum=float(getattr(args, 'proto_bridge_momentum', args.momentum)),
                    use_hard=args.use_hard).cuda()
                proto_bridge_memory.features = bridge_features.cuda()
                proto_bridge_memory_s = ClusterMemory(
                    768, bridge_features_s.size(0), temp=float(getattr(args, 'proto_bridge_temp', args.temp)),
                    momentum=float(getattr(args, 'proto_bridge_momentum', args.momentum)),
                    use_hard=args.use_hard).cuda()
                proto_bridge_memory_s.features = bridge_features_s.cuda()
                trainer.enable_proto_bridge = True
                trainer.proto_bridge_weight = float(getattr(args, 'proto_bridge_weight', 0.05))
                trainer.proto_bridge_label_rgb = bridge_label_rgb.cuda()
                trainer.proto_bridge_label_ir = bridge_label_ir.cuda()
                trainer.proto_bridge_memory = proto_bridge_memory
                trainer.proto_bridge_memory_s = proto_bridge_memory_s
            print('[PBRIDGE] epoch={} active={} pairs={} rgb_cov={:.2%} ir_cov={:.2%} '
                  'rgb_cluster_cov={:.2%} ir_cluster_cov={:.2%} sim(mean/min)={:.4f}/{:.4f} '
                  'margin_mean={:.4f} weight={:.3f} min_sim={:.3f} min_margin={:.3f}'.format(
                      epoch, int(trainer.enable_proto_bridge), int(bridge_stats.get('pairs', 0)),
                      float(bridge_stats.get('rgb_coverage', 0.0)),
                      float(bridge_stats.get('ir_coverage', 0.0)),
                      float(bridge_stats.get('rgb_cluster_coverage', 0.0)),
                      float(bridge_stats.get('ir_cluster_coverage', 0.0)),
                      float(bridge_stats.get('mean_sim', 0.0)),
                      float(bridge_stats.get('min_sim', 0.0)),
                      float(bridge_stats.get('mean_margin', 0.0)),
                      float(getattr(args, 'proto_bridge_weight', 0.05)),
                      float(getattr(args, 'proto_bridge_min_sim', 0.42)),
                      float(getattr(args, 'proto_bridge_min_margin', 0.015))))

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
            # Select best model by mAP/mINP（不用 Rank-1）。
            # best_select_mode:
            # - legacy（默认）：与早期脚本一致，便于跨实验对比 best_score 数值
            #   score = 0.60*all_mAP + 0.40*indoor_mAP + 0.10*all_mINP
            # - full：额外加入 indoor_mINP，并在同分下用 indoor_mAP+indoor_mINP 破平局
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
                        help='SYSU 训练/测试 DataLoader batch（IR/RGB 训练均用此值，显存上限下勿再加大）')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='>1 时每 N 个 iter 做一次 optimizer.step（loss 先除以 N 再 backward），显存仍为 -b；'
                             '每 epoch 的 step 次数变为 iters/N，若要总更新次数与 N=1 接近可同比加大 --epochs')
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
                        help='epoch>=cmlabel 时 RGB 聚类 jaccard 的 k1（源码为 15；可试 12/18）')

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
        help='日志与 checkpoint 目录；默认 ./logs/sysu_train（由 main_worker 根据 log_name 生成）。'
             '网格搜索请显式传入，例如 --logs-dir ./logs/sysu_grid/ls0.15_rc0.05_alt',
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
        help='跨模态邻域：alternating=默认，同 trainers 源码 epoch%%2 交替；rgb2ir=每步仅 RGB→IR；both=每步双向（均作消融）',
    )
    parser.add_argument(
        '--trainer-backend',
        type=str,
        default='source',
        choices=('source',),
        help='训练骨架：固定 source（源码 trainer，指标优先）',
    )
    parser.add_argument(
        '--best-select-mode',
        type=str,
        default='full',
        choices=('legacy', 'full'),
        help='选 model_best：full=默认，含 indoor_mINP 且平局用 indoor_mAP+mINP 破优；legacy=0.6*all+0.4*indoor+0.1*all_mINP（同分亦按 indoor 破优）',
    )
    parser.add_argument(
        '--stage1-best-select-mode',
        type=str,
        default='legacy',
        choices=('legacy', 'full', 'follow'),
        help='选 20model_best（epoch<cmlabel，用于 stage2 warm-start）：legacy=默认更稳；full=与最终一致；follow=跟随 --best-select-mode',
    )
    parser.add_argument('--cmlabel', type=int, default=30,
                        help='stage2 start epoch')
    parser.add_argument('--enable-cglf', action='store_true',
                        help='显式开启 CGLF（metric-first 默认关闭）')
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
    parser.add_argument('--enable-proto-bridge', action='store_true',
                        help='Enable v38 Stage-1 mutual RGB/IR prototype bridge memory.')
    parser.add_argument('--proto-bridge-weight', dest='proto_bridge_weight', type=float, default=0.05,
                        help='Weight for the Stage-1 bridge memory loss.')
    parser.add_argument('--proto-bridge-min-sim', dest='proto_bridge_min_sim', type=float, default=0.42,
                        help='Minimum RGB/IR prototype cosine similarity for bridge matching.')
    parser.add_argument('--proto-bridge-min-margin', dest='proto_bridge_min_margin', type=float, default=0.015,
                        help='Minimum top1-top2 prototype similarity margin for both modalities.')
    parser.add_argument('--proto-bridge-min-cluster-size', dest='proto_bridge_min_cluster_size', type=int, default=4,
                        help='Minimum samples in both RGB and IR clusters before a bridge pair is accepted.')
    parser.add_argument('--proto-bridge-max-pairs', dest='proto_bridge_max_pairs', type=int, default=512,
                        help='Maximum accepted bridge prototype pairs per Stage-1 epoch; <=0 keeps all.')
    parser.add_argument('--proto-bridge-temp', dest='proto_bridge_temp', type=float, default=0.05,
                        help='Temperature for bridge memory.')
    parser.add_argument('--proto-bridge-momentum', dest='proto_bridge_momentum', type=float, default=0.1,
                        help='Feature update momentum for bridge memory.')
    main()
