import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm

import model_io
from dataloader import DepthDataLoader

from utils import RunningAverageDict

sys.path.append(sys.path[0])

from models.mainModel import *


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    log_10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))

    return dict(abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse, rmse_log=rmse_log,
                log_10=log_10, silog=silog, a1=a1, a2=a2, a3=a3)

def predict_tta(model, image, args):

    pred = model(image)[-1].squeeze(1)  # [B, H, W]

    image_flipped = torch.flip(image, dims=[3])
    pred_lr = model(image_flipped)[-1].squeeze(1)  
    pred_lr = torch.flip(pred_lr, dims=[2])       

    final = 0.5 * (pred + pred_lr)
    final = nn.functional.interpolate(final.unsqueeze(1), size=image.shape[-2:], mode='bilinear', align_corners=True)

    return final


def eval(model, test_loader, args, gpus=None):
    device = gpus[0] if gpus else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.colorize:
            color_save_dir = os.path.join(args.save_dir, "color_maps")
            os.makedirs(color_save_dir, exist_ok=True)

    metrics = RunningAverageDict()
    total_invalid = 0

    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader):
            image = batch['image'].to(device)
            gt = batch['depth'].to(device)
            final = predict_tta(model, image, args)

            final_np = final.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            final_np[np.isinf(final_np)] = args.max_depth
            final_np[np.isnan(final_np)] = args.min_depth

            valid_mask = np.logical_and(gt_np > args.min_depth, gt_np < args.max_depth)
            if np.sum(valid_mask) == 0:
                total_invalid += 1
                continue

            scale_factor = np.median(gt_np[valid_mask]) / np.median(final_np[valid_mask])
            final_np *= scale_factor
            final_np = np.clip(final_np, args.min_depth, args.max_depth)

            if args.save_dir:
                factor = 1000 if args.dataset == 'nyu' else 256
                image_path = batch['image_path'][0]
                filename = os.path.basename(image_path)
                impath = os.path.splitext(filename)[0]

                pred_path = os.path.join(args.save_dir, f"{impath}.png")
                Image.fromarray((final_np * factor).astype('uint16')).save(pred_path)
                if args.colorize:
                    vmin = args.min_depth
                    vmax = args.max_depth
                    norm_depth = np.clip((final_np - vmin) / (vmax - vmin), 0, 1)
                    if args.dataset.lower() == "nyu":
                        default_cmap = "jet"
                    else:
                        default_cmap = "jet"
                    cmap_name = getattr(args, "cmap", None) or default_cmap
                    cmap = cm.get_cmap(cmap_name)
                    color_map = cmap(norm_depth)
                    color_image = (color_map[:, :, :3] * 255).astype(np.uint8)

                    if args.dataset == 'nyu':
                        impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '')}"
                        # impath = os.path.splitext(os.path.basename(batch['image_path'][0]))[0]
                    else:
                        dpath = batch['image_path'][0].split('/')
                        impath = dpath[1] + "_" + dpath[-1].split('.')[0]

                    color_path = os.path.join(color_save_dir, f"{impath}_color.png")
                    Image.fromarray(color_image).save(color_path)

            if 'has_valid_depth' in batch and not batch['has_valid_depth']:
                total_invalid += 1
                continue

            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_np.shape
                eval_mask = np.zeros_like(valid_mask)

                if args.garg_crop:
                    eval_mask[int(0.4081 * gt_height):int(0.9919 * gt_height),
                              int(0.0359 * gt_width):int(0.9640 * gt_width)] = 1
                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324 * gt_height):int(0.9135 * gt_height),
                                  int(0.0359 * gt_width):int(0.9640 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1

                valid_mask = np.logical_and(valid_mask, eval_mask)
                if np.sum(valid_mask) == 0:
                    total_invalid += 1
                    continue

            metrics.update(compute_errors(gt_np[valid_mask], final_np[valid_mask]))

    print(f"Total invalid: {total_invalid}")
    print("Metrics:", {k: round(v, 3) for k, v in metrics.get_value().items()})


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

def convert_models_to_fp32(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()

    if isinstance(l, nn.MultiheadAttention):
        for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
            tensor = getattr(l, attr)
            if tensor is not None:
                tensor.data = tensor.data.float()

    for name in ["text_projection", "proj"]:
        if hasattr(l, name):
            attr = getattr(l, name)
            if attr is not None:
                attr.data = attr.data.float()

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Model evaluator', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--n-bins', '--n_bins', default=256, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument('--save-dir', '--save_dir', default=None, type=str, help='Store predictions in folder')
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")

    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset gt")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')

    parser.add_argument('--data_path_eval',
                        default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')
    parser.add_argument('--checkpoint_path', '--checkpoint-path', type=str, required=True,
                        help="checkpoint file to use for prediction")

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--do_kb_crop', help='Use kitti benchmark cropping', action='store_true')

    parser.add_argument('--colorize', action='store_true', help='create color depth')
    parser.add_argument('--cmap', type=str, default='jet', help='jet、viridis）')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # args = parser.parse_args()
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    args.distributed = False
    device = torch.device('cuda:{}'.format(args.gpu))
    test = DepthDataLoader(args, 'online_eval').data

    clipModel, _ = load("RN101", device=args.gpu, jit=False)
    model = TotalModel(clipModel, args.gpu, args.max_depth_eval, args.min_depth_eval, args.n_bins)
    convert_models_to_fp32(model)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    args.checkpoint_path = './checkpoints/.'
    model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    model = model.eval()

    eval(model, test, args, gpus=[device])
