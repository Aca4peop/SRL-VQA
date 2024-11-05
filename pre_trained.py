import argparse
import logging
import os
import time
from datetime import datetime, timedelta
import gc
from tqdm import tqdm
from collections import Counter
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from torchvision import transforms

from config import get_config
from database.VQA.dataset_pre_train import VideoDataset
from lr_scheduler import build_scheduler
from optimizer import build_optimizer

from models.build import build_model
from getVQA import getlogger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from torch.utils.tensorboard import SummaryWriter

DAY = "{0:%Y-%m-%d}".format(datetime.now())
TIMESTAMP = "{0:%H-%M}".format(datetime.now())

writer = SummaryWriter()

best, max_acc_type, max_acc_fps,  = 0, 0.0, 0.0


def accuracy(output, target):
    n = len(target)
    if n == 0:
        return 1e-8
    cnt = 0
    for i in range(n):
        if output[i] == target[i]:
            cnt += 1
    return cnt / n


def trainLoop(config, train_set, model, device, criterion, optimizer, logger, writer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    num_steps = len(train_set)

    epoch_pred, epoch_label = [], []
    epoch_type, epoch_type_tar = [], []
    epoch_fps, epoch_fps_tar = [], []

    totalLoss = 0

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start = time.time()
    end = time.time()

    for idx, data in enumerate(tqdm(train_set)):
        ref, dis, label, fps, dis_type = data['ref'], data[
            'dis'], data['label'], data['fps'], data['type']

        B, N, C, D, H, W = ref.size()
        clip = torch.randint(10000000, (1,)) % N
        ref = ref[:, clip, :, :, :, :].squeeze(1)
        dis = dis[:, clip, :, :, :, :].squeeze(1)

        ref = ref.to(device)
        dis = dis.to(device)
        label = label.view(-1, 1).type(torch.float32).to(device)
        dis_type = dis_type.view(-1).type(torch.long).to(device)
        fps = fps.view(-1).type(torch.long).to(device)

        pred = model(ref, dis)

        pred_label, pred_type, pred_fps = pred[0], pred[1], pred[2]

        loss = config.TRAIN.lambda1 * criterion[0](pred_label, label) \
                + config.TRAIN.lambda2 * criterion[1](pred_type, dis_type) \
                + config.TRAIN.lambda3 * criterion[1](pred_fps, fps)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        totalLoss += loss.item()

        epoch_pred.extend(pred_label.detach().cpu().numpy().reshape(-1))
        epoch_label.extend(label.detach().cpu().numpy().reshape(-1))

        epoch_type.extend(pred_type.argmax(
            dim=1).detach().cpu().numpy().reshape(-1))
        epoch_type_tar.extend(dis_type.detach().cpu().numpy().reshape(-1))

        epoch_fps.extend(pred_fps.argmax(
            dim=1).detach().cpu().numpy().reshape(-1))
        epoch_fps_tar.extend(fps.detach().cpu().numpy().reshape(-1))

        loss_meter.update(loss.item(), label.size(0))
        batch_time.update(time.time() - end)
        norm_meter.update(grad_norm)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    SROCC = spearmanr(epoch_pred, epoch_label)[0]
    PLCC = pearsonr(epoch_pred, epoch_label)[0]
    acc_type = accuracy(epoch_type, epoch_type_tar)
    # acc_type = 0.0
    acc_fps = accuracy(epoch_fps, epoch_fps_tar)

    epoch_time = time.time() - start

    logger.info(
        f"epoch : {epoch}\n \t\ttrain SROCC : {SROCC:.4f}\n \t\tPLCC : {PLCC:.4f}\n \t\tepoch Loss : {totalLoss:.4f}\n \t\tepoch_time {epoch_time:.4f}\n")
    logger.info(
        f"epoch: {epoch}\t train Acc_type :{acc_type:.4f}\n \t\ttrain  Acc_fps: {acc_fps:.4f}")

    writer.add_scalar('Loss/Train', totalLoss, epoch)
    writer.add_scalar('SROCC/Train', SROCC, epoch)
    writer.add_scalar('PLCC/Train', PLCC, epoch)
    writer.add_scalar('Acc_type/Train', acc_type, epoch)
    writer.add_scalar('Acc_fps/Train', acc_fps, epoch)


def testLoop(config, test_set, model, device, criterion, logger, writer, epoch, checkpath, dataset):

    model.eval()
    epoch_pred, epoch_label = [], []
    epoch_type, epoch_type_tar = [], []
    epoch_fps, epoch_fps_tar = [], []

    testLoss = 0
    with torch.no_grad():
        for data in tqdm(test_set):
            ref, dis, label, fps, dis_type = data['ref'], data[
                'dis'], data['label'], data['fps'], data['type']

            B, N, C, D, H, W = ref.size()
            ref = ref.contiguous().view(B * N, C, D, H, W)
            dis = dis.contiguous().view(B * N, C, D, H, W)

            ref = ref.to(device)
            dis = dis.to(device)
            label = label.view(-1, 1).type(torch.float32).to(device)
            dis_type = dis_type.view(-1).type(torch.long).to(device)
            fps = fps.view(-1).type(torch.long).to(device)

            pred = model(ref, dis)

            pred_label, pred_type, pred_fps = pred[0], pred[1], pred[2]

            pred = torch.mean(pred_label.view(B, N), dim=1, keepdims=True)
            pred_type = torch.mean(pred_type.view(
                B, N, -1), dim=1, keepdim=True).view(B, -1)
            pred_fps = torch.mean(pred_fps.view(
                B, N, -1), dim=1, keepdim=True).view(B, -1)

            # PLCC Loss Corner Case
            testLoss = 0.0
            # testLoss += config.TRAIN.lambda1 * criterion[0](pred, label) + \
            #             config.TRAIN.lambda2 * criterion[1](pred_type, dis_type) + \
            #             config.TRAIN.lambda3 * criterion[1](pred_fps, fps)

            epoch_pred.extend(pred.cpu().numpy().reshape(-1))
            epoch_label.extend(label.cpu().numpy().reshape(-1))

            # print(f"pred type : {pred_type.size()}, argmax shape : {pred_type.argmax(dim=1)}")
            # print(f"pred fps : {pred_fps.size()}, argmax shape : {pred_fps.argmax(dim=1)}")

            epoch_type.extend(pred_type.argmax(
                dim=1).cpu().numpy().reshape(-1))
            epoch_type_tar.extend(dis_type.cpu().numpy().reshape(-1))
            epoch_fps.extend(pred_fps.argmax(dim=1).cpu().numpy().reshape(-1))
            epoch_fps_tar.extend(fps.cpu().numpy().reshape(-1))

        SROCC = spearmanr(epoch_pred, epoch_label)[0]
        PLCC = pearsonr(epoch_pred, epoch_label)[0]
        acc_type = accuracy(epoch_type, epoch_type_tar)
        # acc_type = 0.0
        acc_fps = accuracy(epoch_fps, epoch_fps_tar)

        global best, max_acc_type, max_acc_fps
        best = max(best, SROCC)
        max_acc_type = max(max_acc_type, acc_type)
        max_acc_fps = max(max_acc_fps, acc_fps)

        print(
            f"epoch : {epoch}\t Best SROCC : {best:.4f}, SROCC : {SROCC:.4f}, PLCC : {PLCC:.4f}, Test Loss : {testLoss:.4f}")
        logger.info(
            f"epoch: {epoch}\t Best SROCC :{best:.4f}\n \t\ttest : SROCC: {SROCC:.4f}\n \t\tPLCC: {PLCC:.4f}\n \t\ttestLoss : {testLoss:.4f}\n")
        logger.info(
            f"epoch: {epoch}\t Best Acc_type :{max_acc_type:.4f}\n \t\ttest : Acc_type: {acc_type:.4f}\n")
        logger.info(
            f"epoch: {epoch}\t Best Acc_fps :{max_acc_fps:.4f}\n \t\ttest : Acc_fps: {acc_fps:.4f}\n")

        writer.add_scalar('Loss/Test', testLoss, epoch)
        writer.add_scalar('SROCC/Test', SROCC, epoch)
        writer.add_scalar('PLCC/Test', PLCC, epoch)
        writer.add_scalar('Acc_type/Test', acc_type, epoch)
        writer.add_scalar('Acc_fps/Test', acc_fps, epoch)

        if SROCC >= best and SROCC >= config.TEST.BESTSROCC or acc_type >= 0.8:
            best = SROCC
            if not os.path.exists(os.path.join(checkpath, DAY)):
                print(f"checkpoints path : {os.path.join(checkpath, DAY)}")
                os.makedirs(os.path.join(checkpath, DAY))
            torch.save(model.state_dict(), os.path.join(
                checkpath, DAY, TIMESTAMP + '_' + str(best) + '_acc_type=' + str(acc_type) + '.pth'))
            print(f"save checkpoints, Best SROCC is : {best}")


def load_pre_trained(config, path, logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config.MODEL.TYPE)

    # new model dict
    model_dict = model.state_dict()
    # load pre trained model
    if config.MODEL.TYPE != 'pre_train':
        pre_dict = torch.load(path, device)
        pretrained_dict = {k: v for k,
                           v in pre_dict.items() if k in model_dict}
        logger.info(f"Length Of Pre trained model : {len(pretrained_dict)}")
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
        pretrained_dict = {
            k: v for k, v in pretrained_model.items() if 'backbone.' + k in model_dict}
        # overwrite the new model dict
        model_dict.update(pretrained_dict)
        # update the dict to new model
        logger.info(f"length of pretrained dict : {len(pretrained_dict)}")

    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    return model, device


def load_model(logger):
    logger.info("Just Load the Model, WithOut Pre-Trained Weight")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config.MODEL.TYPE)
    model.to(device)
    return model, device


def get_pre_train(cfg, transform_train=None, transform_test=None, batch_train=1, batch_test=1, train_percent=0.8):
    summary_dir = 'runs/pre_train_VQA/' + DAY
    writer = SummaryWriter(summary_dir)
    check_path = './checkpoints/' + 'Pre_Train_VQA/'
    logger = getlogger(path='pre_train')

    channel = 1
    size_x = 224
    size_y = 224
    stride_x = 112
    stride_y = 112

    subj_dataset = './database/VQA/pre_train/youtube8M_0.8_splitByContent.json'

    video_path = '/home2/youtube8M/Clips/dis/yuv/'

    logger.info(f"SUBJ_DATASET: {subj_dataset}")
    batch = {'train': batch_train, 'test': batch_test}
    transform_type = {'train': transform_train, 'test': transform_test}

    video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x,
                                     stride_y, frameWant=cfg.VQA.FRAMEWANT, transform=transform_type[x]) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True,
                                 num_workers=cfg.DATA.NUM_WORKERS, drop_last=False) for x in ['train', 'test']}
    return writer, check_path, dataloaders['train'], dataloaders['test'], logger


def main(config):

    dataset = config.DATA.DATASET

    prefix = os.path.abspath('.')
    pre_trained_path = os.path.join(prefix, config.TRAIN.PRE_TRAINED)

    epoch = config.TRAIN.EPOCHS
    batch_train = config.DATA.BATCH_SIZE
    batch_test = config.DATA.BATCH_TEST


    criterion = [PlccLoss(), nn.CrossEntropyLoss()]

    img_size = config.DATA.IMG_SIZE

    transform_train, transform_test = transforms.CenterCrop(
        (256, 256)), transforms.CenterCrop((256, 256))

    writer, check_path, train_set, test_set, logger = \
        get_pre_train(config, transform_train=transform_train, transform_test=transform_test,
                      batch_train=batch_train, batch_test=batch_test)

    if config.TRAIN.USEPRETRAINED:
        print(config.TRAIN.USEPRETRAINED)
        print(config.FINE_TUNE)
        model, device = load_pre_trained(config, pre_trained_path, logger)
    else:
        model, device = load_model(logger)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)

    logger.info(f"number of params: {n_parameters}")

    optimizer = build_optimizer(config, model)

    lr_scheduler = build_scheduler(config, optimizer, len(train_set))

    # print config
    logger.info(model)

    logger.info(config.dump())

    for i in range(epoch):
        trainLoop(config, train_set, model, device, criterion, optimizer,
                  logger, writer, epoch=i, lr_scheduler=lr_scheduler)
        testLoop(config, test_set, model, device, criterion, logger,
                 writer, epoch=i, checkpath=check_path, dataset=dataset)
        gc.collect()

    if not os.path.exists(os.path.join(
            check_path, DAY, TIMESTAMP + '_' + str(best) + '_last.pth')):
        print(
            f"checkpoints path: {os.path.join(check_path, DAY, TIMESTAMP + '_' + str(best) + '_last.pth')}")
        os.makedirs(os.path.join(check_path, DAY))
    torch.save(model.state_dict(), os.path.join(
        check_path, DAY, TIMESTAMP + '_' + str(best) + '_last.pth'))

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
        val = 1.0 - plcc(pred.view(-1).float(), target.view(-1).float())
        return torch.log(val)


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

    parser.add_argument('--pretrained', help='wether use pretrained model')
    parser.add_argument('--dataset', type=str,
                        default='pre_train', help='dataset')
    parser.add_argument('--dataset_test', type=str,
                        default='CSIQ_VQA', help='dataset')
    parser.add_argument('--model', type=str,
                        default='pre_train', help='Model Type')
    parser.add_argument('--frame', type=int, default='100',
                        help='Frame Per Video')
    parser.add_argument('--base_lr', type=float,
                        default='3e-4', help='Base Learning Rate')
    parser.add_argument('--loss', type=str, default='plcc',
                        help='Loss Function')
    parser.add_argument('--best', type=float, default='0.65',
                        help='Best SROCC For Save Checkpoints')
    parser.add_argument('--epoch', type=int,
                        default='100', help='Epoch Number')
    parser.add_argument('--warm_up_epochs', type=int,
                        default='4', help='Warm Up Epoch Number')
    parser.add_argument('--pre_trained_path', type=str,
                        default='./checkpoints/Pre_Train_VQA/Best/23-12_0.8186017673847709.pth', help='pretrained weight path')
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

    main(config)
