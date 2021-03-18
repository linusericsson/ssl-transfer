# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset_generic import Dataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm
from io_utils import save_pickle, load_pickle
from normals_evaluation_utils import NormalsEvaluator
import sys


colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, args):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    cv2.imwrite(os.path.join(args.result,
                img_name.replace('.jpg', '.png')), im_vis)


def evaluate_surface_normals(segmentation_module, loader, args):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()
    dataset = loader.dataset
    vocab_file = dataset.vocab_file
    print('Loading normals')
    img_id_to_normals_data = dataset.load_normals_file()

    segmentation_module.eval()
    normals_evaluator = NormalsEvaluator(vocab_file=vocab_file)
    print('Start evaluation of %d batches' % (len(loader)))

    pbar = tqdm(total=len(loader))
    accum_probs = []
    accum_gt = []
    accum_masks = []
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        img_id = batch_data['info']
        raw_normals, valid_mask = img_id_to_normals_data(img_id)

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            if len(img_resized_list) > 1:
                scores = torch.zeros(1, dataset.num_classes, segSize[0], segSize[1])
                scores = async_copy_to(scores, args.gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, args.gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                if len(img_resized_list) > 1:
                    scores = scores + scores_tmp / len(args.imgSize)
                else:
                    scores = scores_tmp / len(args.imgSize)

            probs = scores
            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            prob_np = as_numpy(probs)

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, dataset.num_classes)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # accumulate probs for normal
        # since we compute median stats, we cannot average over images
        accum_probs.append(prob_np.squeeze())
        accum_gt.append(raw_normals)
        accum_masks.append(valid_mask.astype(np.int32))

        # visualization
        if args.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred, args)

        pbar.update(1)

    # compute normal metrics
    print("Computing pixelwise normal metrics")
    metrics = normals_evaluator.compute_normals_metrics_from_network_probs(accum_probs, accum_gt, accum_masks)
    normals_evaluator.print_metrics(metrics)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}%, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean() * 100, acc_meter.average() * 100, time_meter.average()))
    return metrics


def evaluate_segmentation(segmentation_module, loader, args):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()
    dataset = loader.dataset

    segmentation_module.eval()
    print('Start evaluation of %d batches' % (len(loader)))

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, dataset.num_classes, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, args.gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(args.imgSize)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, dataset.num_classes)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        if args.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred, args)

        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    per_class_iou = []
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))
        per_class_iou.append(_iou)

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}%, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean() * 100, acc_meter.average() * 100, time_meter.average()))
    metrics = {
        'mean_iou': float(iou.mean()),
        'acc': float(acc_meter.average()),
        'per_class_iou': per_class_iou,
    }
    return metrics


def main(args):
    torch.cuda.set_device(args.gpu)

    crit = nn.NLLLoss(ignore_index=-1)

    # Dataset and Loader
    dataset_val = Dataset(
        args, split_name=args.split_name,
        batch_per_gpu=args.batchsize)
    loader_val = torchdata.DataLoader(
        dataset_val,
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    # Network Builders
    builder = ModelBuilder()
    print('Loading encoder from: %s' % (args.weights_encoder))
    print('Loading decoder from: %s' % (args.weights_decoder))
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=dataset_val.num_classes,
        weights=args.weights_decoder,
        use_softmax=True)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    segmentation_module.cuda()

    # Main loop
    if args.dataset.startswith('nyuv2sn'):
        metrics = evaluate_surface_normals(segmentation_module, loader_val, args)
    else:
        metrics = evaluate_segmentation(segmentation_module, loader_val, args)

    save_pickle(args.result_file, metrics)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--dirname', required=True,
                        help="where checkpoints are stored")
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")
    parser.add_argument('--image_mode', required=True,
                        choices=['rgb', 'bgr', 'lab', 'rgb_nonorm'])
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')
    parser.add_argument('--dataset', required=True,
                        type=str, choices=['ade', 'nyuv2sn20', 'nyuv2sn40'])
    parser.add_argument('--split_name', required=True,
                        choices=['val', 'test', 'train'])


    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--imgSize', default=[450], nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g.  300 400 500 600')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')

    # Misc arguments
    parser.add_argument('--visualize', action='store_true',
                        help='output visualization?')
    parser.add_argument('--result', default='./result',
                        help='folder to output visualization results')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id for evaluation')

    args = parser.parse_args()
    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()

    # absolute paths of model weights
    # find a directory
    model_id_dir = os.listdir(args.dirname)
    if len(model_id_dir) != 0 and \
       os.path.isdir(os.path.join(args.dirname, model_id_dir[0])):
        model_dir = os.path.join(args.dirname, model_id_dir[0])
    else:
        model_dir = args.dirname

    args.batchsize = 1 # only supports batchsize == 1, single GPU

    args.weights_encoder = os.path.join(model_dir,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(model_dir,
                                        'decoder' + args.suffix)
    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exist!'

    args.result = os.path.join(args.dirname, 'results_' + args.split_name)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)
    suffix = args.suffix.split('.')[0]
    args.result_file = os.path.join(args.result, 'results_%s.pkl' % (suffix))
    if os.path.isfile(args.result_file):
        print('Found file %s. Skipping' % (args.result_file))
        sys.exit(0)
    result_lock_file = os.path.join(args.result, 'results_%s.pkl.lock' % (suffix))
    if os.path.isdir(result_lock_file):
        print('Found lock file %s. Skipping' % (args.result_file))
        sys.exit(0)
    os.makedirs(result_lock_file)

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    main(args)
    os.rmdir(result_lock_file)
