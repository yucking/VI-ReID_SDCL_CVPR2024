# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
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
from clustercontrast.trainers import ClusterContrastTrainer_SDCL
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
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,Gray
from solver.scheduler_factory import create_scheduler
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import os.path as osp
import errno

start_epoch = best_mAP = 0
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
part=1
torch.backends.cudnn.enable =True,
torch.backends.cudnn.benchmark = True


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

def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
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
        if self.transform is not None:
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
        cudnn.deterministic = True
    log_name = 'sysu_train'
    main_worker_stage2(args,log_name)

def main_worker_stage2(args,log_name):
    
    # l2norm = Normalize(2) # 创建了一个 L2 归一化层，用于后续处理数据时的归一化操作
    ir_batch=args.batch_size
    rgb_batch=args.batch_size

    global start_epoch, best_mAP 

    args.logs_dir = osp.join('./logs',log_name)
    start_time = time.monotonic()

    cudnn.benchmark = True # 自主选择最优算法？ 
    
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

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)

    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0) # transformer model
    # 将模型移动到GPU上，并使用 nn.DataParallel 包装模型以支持多GPU训练。
    model.cuda()
    model = nn.DataParallel(model)
    # 创建一个训练器对象，设置训练器的一些参数
    trainer = ClusterContrastTrainer_SDCL(model)
    trainer.cmlabel=30 #30
    trainer.hm = 0 #20 sysu_release_k1_10_ins16 设置的1
    trainer.ht = 10 #10#10# 
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

    # 开始训练循环
    for epoch in range(args.epochs):
        if (epoch == trainer.cmlabel) :# 30 在特定轮次执行特定操作
            # 加载之前保存的最佳模型检查点
            checkpoint = load_checkpoint(osp.join(args.logs_dir,'20model_best.pth.tar'))
            model.load_state_dict(checkpoint['state_dict'])
        
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            # features: deep/global embedding（来自 transformer tokens 经 b1 + CLS + BN，且做了 flip-avg）
            # features_s: shallow embedding（由 TransReID 在 blocks 前取 patch tokens x[:,1:]，经 b2 + CLS + BN，且做了 flip-avg）
            features_rgb, features_rgb_s = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)    # 特征已经是按顺序排列的
            features_rgb_s = torch.cat([features_rgb_s[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0) # 浅层图像特征拼接
            features_rgb_ori = features_rgb
            features_rgb_ori_ = F.normalize(features_rgb_ori, dim=1)
            features_rgb_s_ = F.normalize(features_rgb_s, dim=1)
            features_rgb = torch.cat((features_rgb,features_rgb_s), 1)
            features_rgb_ = F.normalize(features_rgb, dim=1) # 聚类用的是 深层 + 浅层拼接后的特征，更健壮

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
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
            
            # Jaccard 距离 + DBSCAN 聚类
            ir_eps = 0.6 # 0.6
            rgb_eps = 0.6 # 0.6#+0.1
            print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
            cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
            print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
            cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)
            
            rerank_dist_ir = compute_jaccard_distance(features_ir_, k1=30, k2=args.k2,search_option=3)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            if epoch >= trainer.cmlabel:
                args.k1 = 15 #15 #10
                iters = 100
            rerank_dist_rgb = compute_jaccard_distance(features_rgb_, k1=args.k1, k2=args.k2,search_option=3) # 距离矩阵
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb) # 伪标签
            del rerank_dist_rgb
            del rerank_dist_ir
            pseudo_labels_all = []
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
        
        # 生成聚类特征和聚类内存
        '''tensor([
                [8., 2., 0., 0.],  # 第 0 行 = 簇 0 的中心
                [0., 0., 40., 0.]   # 第 1 行 = 簇 1 的中心])'''
        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir_ori) # 根据伪标签和原始特征生成聚类特征
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb_ori) 
        memory_ir = ClusterMemory(768, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(768, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb
        
        wise_momentum=0.9
        print('wise_momentum',wise_momentum)
        # 实例级别的内存 （原图归一化特征）
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
        trainer.nameMap_ir = nameMap_ir
        trainer.nameMap_rgb = nameMap_rgb


######## 
        # 生成聚类特征和聚类内存(浅层)
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

        wise_memory_rgb_s = Memory_wise_v3(768, len(dataset_rgb.train),num_cluster_rgb,temp=args.temp, momentum=wise_momentum).cuda()#0.9
        wise_memory_ir_s = Memory_wise_v3(768, len(dataset_ir.train),num_cluster_ir,temp=args.temp, momentum=wise_momentum).cuda()#args.momentum
        wise_memory_ir_s.features = F.normalize(features_ir_s, dim=1).cuda()
        wise_memory_rgb_s.features = F.normalize(features_rgb_s, dim=1).cuda()
        trainer.wise_memory_ir_s = wise_memory_ir_s
        trainer.wise_memory_rgb_s = wise_memory_rgb_s


        # 根据训练集及其相应的伪标签，分理处正常与异常标签集
        pseudo_labeled_dataset_ir = []
        ir_label=[]
        pseudo_real_ir = {}
        cams_ir = []
        modality_ir = []
        outlier=0
        cross_cam=[]
        ir_cluster=collections.defaultdict(list)
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            cams_ir.append(cid)
            modality_ir.append(1)
            cross_cam.append(int(cid+4)) # 
            ir_label.append(label.item())
            ir_cluster[cid].append(label.item())
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid)) 
                pseudo_real_ir[label.item()] = pseudo_real_ir.get(label.item(),[])+[_]
                pseudo_real_ir[label.item()] = list(set(pseudo_real_ir[label.item()]))
            else:
                outlier=outlier+1
        print('==> Statistics for IR epoch {}: {} clusters, outlier {}'.format(epoch, num_cluster_ir,outlier))

        pseudo_labeled_dataset_rgb = [] # 伪标签数据集
        rgb_label=[]    # 所有的伪标签（包括异常标签 -1）
        pseudo_real_rgb = {} # 映射“伪标签”到“原始真实 ID”
        cams_rgb = [] # 所有图像的摄像机 ID 列表
        modality_rgb = [] # 所有图像的模态列表（RGB 模态为 0）
        outlier = 0 # 异常标签计数器
        rgb_cluster = collections.defaultdict(list) # 用于存储每个摄像机有多少个簇 列表

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            cams_rgb.append(cid)
            modality_rgb.append(0)
            cross_cam.append(int(cid))
            rgb_label.append(label.item())
            rgb_cluster[cid].append(label.item())
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                
                pseudo_real_rgb[label.item()] = pseudo_real_rgb.get(label.item(),[])+[_]
                pseudo_real_rgb[label.item()] = list(set(pseudo_real_rgb[label.item()])) # 伪标签ID：对应的原始图像ID列表
            else:
                outlier=outlier+1
        print('==> Statistics for RGB epoch {}: {} clusters, outlier {} '.format(epoch, num_cluster_rgb,outlier))

#### Stage 2 Collaborative Ranking Association
        if epoch >= trainer.cmlabel:
            with torch.no_grad():
                # =========================================================
                # WCRA-RC: Weighted CRA + Reciprocal Check
                # 保持单向 RGB -> IR 写回，不改 trainer，不改 shared memory 结构
                # =========================================================
                TOPK2 = 20
                Score_TOPK = 20
                REV_TOPM = 8          # 弱双向互检：IR->RGB 只做互检，不做反向写标签
                SMOOTH_K = 5          # 与原论文/源码保持一致的 visible 内平滑邻居数

                # 加权参数
                ALPHA_D = 0.6         # deep 排名权重
                ALPHA_S = 0.4         # shallow 排名权重
                BETA_REC = 0.35       # reciprocal bonus 权重
                SMOOTH_TEMP = 0.07    # weighted smoothing 温度
                MIN_FORWARD_SCORE = 1e-8

                # ---------- 1) RGB -> IR 深浅层 top-k 排名 ----------
                cluster_label_ir_self = trainer.wise_memory_ir.labels.detach().cpu()   # [N_ir]

                # deep similarity: RGB x IR
                ins_sim_rgb_ir = features_rgb_ori_.mm(features_ir_ori_.t())             # [N_rgb, N_ir]
                topk_val_d, topk_idx_d = torch.topk(ins_sim_rgb_ir, int(Score_TOPK), dim=1)

                # shallow similarity: RGB x IR
                ins_sim_rgb_ir_s = features_rgb_s_.mm(features_ir_s_.t())               # [N_rgb, N_ir]
                topk_val_s, topk_idx_s = torch.topk(ins_sim_rgb_ir_s, int(Score_TOPK), dim=1)

                # ---------- 2) 弱双向互检：IR -> RGB top-m ----------
                # 只用来给 forward 候选加 bonus，不做 IR 标签重写
                _, ir2rgb_idx_d = torch.topk(ins_sim_rgb_ir.t(), int(REV_TOPM), dim=1)      # [N_ir, M]
                _, ir2rgb_idx_s = torch.topk(ins_sim_rgb_ir_s.t(), int(REV_TOPM), dim=1)    # [N_ir, M]

                # 后续都在 CPU 上做，省显存
                topk_val_d = topk_val_d.detach().cpu()
                topk_idx_d = topk_idx_d.detach().cpu()
                topk_val_s = topk_val_s.detach().cpu()
                topk_idx_s = topk_idx_s.detach().cpu()
                ir2rgb_idx_d = ir2rgb_idx_d.detach().cpu()
                ir2rgb_idx_s = ir2rgb_idx_s.detach().cpu()

                # 每个 RGB 对应 top-k IR 的伪标签
                label_topk_d = cluster_label_ir_self[topk_idx_d]   # [N_rgb, K]
                label_topk_s = cluster_label_ir_self[topk_idx_s]   # [N_rgb, K]

                # ---------- 3) 保留原 CRA 的硬交集结果，作为 fallback ----------
                intersect_count_list = []
                for l in range(TOPK2):
                    intersect_count = (
                        label_topk_s == label_topk_d[:, l].view(-1, 1)
                    ).int().sum(1).view(-1, 1)
                    intersect_count_list.append(intersect_count)

                intersect_count_list = torch.cat(intersect_count_list, 1)                # [N_rgb, K]
                _, hard_label_index = torch.topk(intersect_count_list, 1, dim=1)

                hard_rgb_ir_label = torch.gather(
                    label_topk_d, dim=1, index=hard_label_index.view(-1, 1)
                ).view(-1).numpy().astype(np.int64)                                      # [N_rgb]

                # ---------- 4) 位置加权 + reciprocal bonus 的 forward 打分 ----------
                num_rgb = topk_idx_d.size(0)
                num_ir_cls = int(num_cluster_ir)

                # 排名位置权重：前面的名次更重要
                rank_w = torch.tensor(
                    [1.0 / np.log2(k + 2.0) for k in range(Score_TOPK)],
                    dtype=torch.float32
                )  # [K]

                # reciprocal: 若某个 IR 邻居也把当前 RGB 放进了自己的 top-m，则给 bonus
                rgb_ids = torch.arange(num_rgb).view(-1, 1, 1)

                rec_d = (ir2rgb_idx_d[topk_idx_d] == rgb_ids).any(dim=2).float()         # [N_rgb, K]
                rec_s = (ir2rgb_idx_s[topk_idx_s] == rgb_ids).any(dim=2).float()         # [N_rgb, K]

                # 建一个 [N_rgb, num_cluster_ir] 的标签得分表
                score_table = torch.zeros((num_rgb, num_ir_cls), dtype=torch.float32)

                for k in range(Score_TOPK):
                    # deep 分支
                    lb_d = label_topk_d[:, k]                                            # [N_rgb]
                    valid_d = (lb_d >= 0)
                    if valid_d.any():
                        sim_factor_d = ((topk_val_d[:, k] + 1.0) / 2.0).clamp(min=0.0)   # [-1,1] -> [0,1]
                        bonus_d = 1.0 + BETA_REC * rec_d[:, k]
                        w_d = ALPHA_D * rank_w[k] * sim_factor_d * bonus_d
                        score_table[valid_d, lb_d[valid_d].long()] += w_d[valid_d]

                    # shallow 分支
                    lb_s = label_topk_s[:, k]
                    valid_s = (lb_s >= 0)
                    if valid_s.any():
                        sim_factor_s = ((topk_val_s[:, k] + 1.0) / 2.0).clamp(min=0.0)
                        bonus_s = 1.0 + BETA_REC * rec_s[:, k]
                        w_s = ALPHA_S * rank_w[k] * sim_factor_s * bonus_s
                        score_table[valid_s, lb_s[valid_s].long()] += w_s[valid_s]

                forward_best_score, forward_best_label = torch.max(score_table, dim=1)   # [N_rgb], [N_rgb]
                forward_best_score = forward_best_score.numpy()
                forward_best_label = forward_best_label.numpy().astype(np.int64)

                # 若该样本没有有效 forward score，则 fallback 到原 CRA 的硬计数结果
                fallback_mask = forward_best_score <= MIN_FORWARD_SCORE
                forward_best_label[fallback_mask] = hard_rgb_ir_label[fallback_mask]

                # ---------- 5) visible 内 weighted smoothing ----------
                print('WCRA-RC: weighted rank fusion + reciprocal check + weighted smoothing')

                # one-hot，保留 0 号位作为无效/噪声
                rgb_cm_label = torch.from_numpy(forward_best_label).view(-1) + 1          # [N_rgb]
                rgb_cm_label = F.one_hot(
                    rgb_cm_label.view(num_rgb, 1).long(),
                    num_ir_cls + 1
                ).float().squeeze(1)                                                      # [N_rgb, num_ir_cls + 1]

                # visible 内 shallow + deep 相似度
                rgb_self_sim = torch.mm(features_rgb_ori_, features_rgb_ori_.t())
                rgb_self_sim_s = torch.mm(features_rgb_s_, features_rgb_s_.t())
                rgb_self_sim = (rgb_self_sim + rgb_self_sim_s).detach().cpu()             # [N_rgb, N_rgb]

                # top-k 邻居
                topk_self_val, topk_self_idx = torch.topk(rgb_self_sim, SMOOTH_K, dim=1)  # [N_rgb, K]
                topk_self_w = F.softmax(topk_self_val / SMOOTH_TEMP, dim=1)               # [N_rgb, K]

                # 不建巨大的 Pvv 稠密矩阵，直接 gather 邻居 one-hot 后加权求和
                neighbor_label_prob = rgb_cm_label[topk_self_idx]                          # [N_rgb, K, C]
                smooth_score = (topk_self_w.unsqueeze(2) * neighbor_label_prob).sum(dim=1)# [N_rgb, C]

                smooth_top2_score, smooth_top2_idx = torch.topk(smooth_score, k=2, dim=1)
                pseudo_labels_rgb_cm = (smooth_top2_idx[:, 0].numpy() - 1).astype(np.int64)

                # 若 smoothing 选到了无效类（0 -> -1），则回退到 forward_best_label
                invalid_smooth_mask = pseudo_labels_rgb_cm < 0
                pseudo_labels_rgb_cm[invalid_smooth_mask] = forward_best_label[invalid_smooth_mask]

                # 统计信息
                smooth_margin = (smooth_top2_score[:, 0] - smooth_top2_score[:, 1]).numpy()
                print(
                    '[WCRA-RC] epoch={} score(mean/p50/p90)={:.4f}/{:.4f}/{:.4f} '
                    'margin(mean/p50/p90)={:.4f}/{:.4f}/{:.4f}'.format(
                        epoch,
                        float(np.mean(forward_best_score)),
                        float(np.percentile(forward_best_score, 50)),
                        float(np.percentile(forward_best_score, 90)),
                        float(np.mean(smooth_margin)),
                        float(np.percentile(smooth_margin, 50)),
                        float(np.percentile(smooth_margin, 90)),
                    )
                )

                cluster_label_rgb_ir = torch.from_numpy(pseudo_labels_rgb_cm)

                del rgb_self_sim, rgb_self_sim_s, smooth_score

            '''通过'''
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

            cluster_features_ir = generate_cluster_features(pseudo_labels_all, features_all) # 生成聚类特征

############## 共享层
            shared_memory = ClusterMemory(768, num_cluster_ir, temp=args.temp,
                                   momentum=0.1, use_hard=args.use_hard)#.cuda() 聚类内存
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


        #########伪标签精炼结束 

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

            args.test_batch=256
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
            is_best = (cmc[0] > best_mAP)
            best_mAP = max(cmc[0], best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': cmc[0],
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            if epoch < trainer.cmlabel:
                save_checkpoint10({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': cmc[0],
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))


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


            print('\n * Finished epoch {:3d}  model r1: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, cmc[0], best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()
        # if epoch >30:
        #     break
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
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
    parser.add_argument('-b', '--batch-size', type=int, default=2)
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

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        )
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
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
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    parser.add_argument('--warmup-step', type=int, default=0)
    parser.add_argument('--milestones', nargs='+', type=int, default=[20,40],
                        help='milestones for the learning rate decay')


    main()
