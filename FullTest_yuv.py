import argparse
import logging
import os
import time
from datetime import datetime, timedelta
import gc
from typing import Collection
from tqdm import tqdm

import scipy.io as sio

import numpy as np
import torch 
import torch.backends.cudnn as cudnn
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, kendalltau
from timm.utils import AverageMeter

from config import get_config
# from database.VQA.dataset_yuv import VideoDataset
from lr_scheduler import build_scheduler
from optimizer import build_optimizer

from models.build import build_model
from getVQA import *
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

# from models.build_tsm import build_model
# from models.build_cnn_swin_tsm import build_model
# from models.build_cnn_swin import build_model
# from models.build_ape import build_model

bestSROCC = 0
bestPLCC = 0
bestRMSE = 1e9
bestKROCC = 0

def testLoop(config, test_set, model, device, logger, epoch, disNum):
    
    model.eval()
    
    epoch_pred, epoch_label, epoch_dis = [], [], []


    testLoss = 0
    with torch.no_grad():
        for (img, label) in tqdm(test_set):
            # img, label = data["img"], data["label"]
            # img = img.view(-1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
            # img = img.squeeze(1)

            B, N, C, D, H, W = img.size()
            # img = img.view(B, N * C, D, H, W)
            img = img.contiguous().view(B * N, C, D, H, W)
            # B, C, D, H, W = img.size()

            # TODO : FIVE CROP
            # B, C, D, H, W = img.shape 
            # img = img.view(B * C, 1, D, H, W)

            img = img.to(device)
            label = label.view(-1, 1).to(device)
            pred = model(img)
            pred = torch.mean(pred.view(B, N), dim=1, keepdims=True)
            
            epoch_label.extend(label.cpu().numpy().reshape(-1))
            epoch_pred.extend(pred.cpu().numpy().reshape(-1))


        mdict = {'pred' : epoch_pred, 'label' : epoch_label}
        sio.savemat(config.DATA.DATASET + 'test.mat', mdict) 
        
        SROCC = spearmanr(epoch_label, epoch_pred)[0]
        PLCC = pearsonr(epoch_pred, epoch_label)[0]
        RMSE = np.sqrt(((np.array(epoch_pred) - np.array(epoch_label)) ** 2).mean())
        KROCC = kendalltau(epoch_pred, epoch_label)[0]


        global bestSROCC, bestPLCC, bestRMSE, bestKROCC
        bestSROCC = max(bestSROCC, SROCC)
        bestPLCC = max(bestPLCC, PLCC)
        bestKROCC = max(bestKROCC, KROCC)
        bestRMSE = min(bestRMSE, RMSE)

        print(f"epoch : {epoch}\t Best SROCC : {bestSROCC:.4f} Best PLCC :{bestPLCC:.4f} SROCC : {SROCC:.4f}, PLCC : {PLCC:.4f}, Test Loss : {testLoss:.4f}")
        logger.info(f"epoch: {epoch}\t Best SROCC :{bestSROCC:.4f} \tBest PLCC :{bestPLCC:.4f}\tBest KROCC :{bestKROCC:.4f}\tBest RMSE :{bestRMSE:.4f} \n \t\ttest : SROCC: {SROCC:.4f}\n \t\tPLCC: {PLCC:.4f}\n \t\tKROCC: {KROCC:.4f}\n \t\tRMSE: {RMSE:.4f}\n")



def load_pre_trained(logger, config, path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = SwinTransformer()
    model = build_model(config.MODEL.TYPE)

    # new model dict
    model_dict = model.state_dict()
    # load pre trained model
    if config.MODEL.TYPE != 'pre_train':
        pre_dict = torch.load(path, device)
        pretrained_dict ={k : v for k, v in pre_dict.items() if k in model_dict}
        logger.info(f"Model Type : {config.MODEL.TYPE}\t Length Of Pre trained model : {len(pretrained_dict)}")
        model_dict.update(pretrained_dict)
    else:
        pretrained_model = torch.load(path, device)['model']
        if 'head.weight' in pretrained_model:
            pretrained_model.pop('head.weight')
        if 'head.bias' in pretrained_model:
            pretrained_model.pop('head.bias')
        # get the same weight
        # pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
        # TODO VB_SWIN
        pretrained_dict = {k: v for k, v in pretrained_model.items() if 'backbone.' + k in model_dict}
        # overwrite the new model dict
        model_dict.update(pretrained_dict)
        # update the dict to new model
        logger.info(f"Model Type : {config.MODEL.TYPE}\t Length Of Pre trained model : {len(pretrained_dict)}")
        print(f"length of pretrained dict : {len(pretrained_dict)}")

    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    return model, device


def load_model(logger):
    logger.info("Just load model, WithOut Pre-Trained Weight")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config.MODEL.TYPE)
    model.to(device)
    return model, device


def main(config, idx):

    dataset = config.DATA.DATASET

    epoch = 1
    batch_train = 8
    batch_test = 1


    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    img_size = config.DATA.IMG_SIZE

    # transform_train, transform_test = get_transforms(img_size1=540, img_size2=config.DATA.IMG_SIZE)
    # transform_train, transform_test = get_transforms(img_size1=540, img_size2=512)
    transform_train, transform_test = None, None


    if dataset == 'LIVE_IQA':
        writer, check_path, train_set, test_set, logger = \
            getLIVE(transform_train, transform_test,\
                    batch_train=batch_train, batch_test=batch_test)
    elif dataset == 'KON_IQA':
        writer, check_path, train_set, test_set, logger = \
            getKonIQ(transform_train, transform_test, batch_train=batch_train, batch_test=batch_test, train_percent=config.TRAIN.PERCENT)
    elif dataset == 'LIVE_VQA':
        writer, check_path, train_set, test_set, logger = \
            getLIVEVQA(config, transform_train, transform_test, batch_train=batch_train, batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'KON_VQA':
        writer, check_path, train_set, test_set, logger = \
            getKonVQA(config, transform_train, transform_test, batch_train=batch_train, batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'CSIQ_VQA':
        writer, check_path, train_set, test_set, logger = \
            getCSIQVQA(config, transform_train, transform_test, batch_train=batch_train, batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'CVD_VQA':
        writer, check_path, train_set, test_set, logger = \
            getCVDVQA(config, transform_train, transform_test, batch_train=batch_train, batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'LIVEM_VQA':
        writer, check_path, train_set, test_set, logger = \
            getLIVEMVQA(config, transform_train, transform_test, batch_train=batch_train, batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'VQC_VQA':
        writer, check_path, train_set, test_set, logger = \
            getVQCVQA(config, transform_train, transform_test, batch_train=batch_train, batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)
    
    prefix = os.path.abspath('.')
    # pre_trained_path = os.path.join(prefix, 'pre_trained/swin_tiny_patch4_window7_224.pth')
    pre_trained_path = os.path.join(prefix, config.TRAIN.PRE_TRAINED)

    if config.FINE_TUNE:
        model, device = load_pre_trained(logger, config, pre_trained_path)
    else:
        model, device = load_model(logger)  

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"number of params: {n_parameters}")

    logger.info(f"PRE TRAINED PATH : {config.TRAIN.PRE_TRAINED}")
    # print config
    logger.info(model)

    logger.info(config.dump())

    global bestSROCC
    global bestPLCC
    bestSROCC = 0
    bestPLCC = 0

    disNum = 4
    if dataset == "CSIQ_VQA":
        disNum = 6

    for i in range(epoch):
        testLoop(config, test_set, model, device, logger, epoch=i, disNum=disNum)
        gc.collect()

    writer.close()
    print("Done!")


def plcc(pred, target):
    n = len(pred)
    if n != len(target):
        raise ValueError('input and target must have the same length')
    if n < 2:
        raise ValueError('input length must greater than 2')

    xmean = torch.mean(pred)
    ymean = torch.mean(target)
    xm = pred - xmean
    ym = target - ymean
    bias = 1e-8
    normxm = torch.norm(xm, p=2) + bias
    normym = torch.norm(ym, p=2) + bias

    r = torch.dot(xm/normxm, ym/normym)
    return r

class PlccLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # print(pred.shape, target.shape)
        val = 1.0 - plcc(pred.view(-1).float(), target.view(-1).float())
        return torch.log(val)
        # return 1.0 - plcc(pred.view(-1).float(), target.view(-1).float())

def parse_option():
    parser = argparse.ArgumentParser(
        'Self-Supervised Representation Learning for Video Quality Assessment training and evaluation script', add_help=False)

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--batch-test', type=int, default=24,
                        help="batch test size for single GPU")
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--dataset', type=str,
                        default='LIVE_VQA', help='dataset')
    parser.add_argument('--dataset_test', type=str,
                        default='CSIQ_VQA', help='dataset')
    parser.add_argument('--model', type=str,
                        default='vb_cnn_transformer', help='Model Type')
    parser.add_argument('--frame', type=int, default='100',
                        help='Frame Per Video')
    parser.add_argument('--base_lr', type=float,
                        default='5e-5', help='Base Learning Rate')
    parser.add_argument('--loss', type=str, default='plcc',
                        help='Loss Function')
    parser.add_argument('--best', type=float, default='0.78',
                        help='Best SROCC For Save Checkpoints')
    parser.add_argument('--epoch', type=int,
                        default='100', help='Epoch Number')
    parser.add_argument('--warm_up_epochs', type=int,
                        default='5', help='Warm Up Epoch Number')

    parser.add_argument('--pre_trained_path', type=str, default='./pretrained/pre_trained.pth', help='pretrained weight path')

    parser.add_argument('--fine_tune', type=bool, help='Fine Tune Or Not')
    parser.add_argument('--five', type=bool,
                        help='FIVE Test Calc Mean-Std Result.')
    parser.add_argument('--idx', type=int, default=-1, help='Index')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


if __name__ == '__main__':

    _, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE
    linear_scaled_lr = config.TRAIN.BASE_LR
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR
    linear_scaled_min_lr = config.TRAIN.MIN_LR

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        print("USING ACCUMULATION_STEPS")
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    print("base lr :", linear_scaled_lr)
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    print("warmup lr :", linear_scaled_warmup_lr)
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    print("min lr :", linear_scaled_min_lr)
    config.freeze()

    if config.FIVE:
        for idx in range(5):
            main(config, idx)
    else:
        main(config, 100)
