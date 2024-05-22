from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import numpy.ma as ma

import torch
from torch.utils.data import DataLoader
import scipy.stats
from layers import disp_to_depth, transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import matplotlib as mpl
import matplotlib.cm as cm
from evaluate_pose import dump_xyz, compute_ate
from matplotlib import pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def calculate_pixelwise_nonconformity_scores(y_pred, gt_depth, min_depth=1e-3, max_depth=80, split="eigen"):
    gt_height, gt_width = gt_depth.shape[:2]
    if split == "eigen" or "seq" in split:
        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
        
    nonconformity_scores = np.abs(gt_depth - y_pred) * mask  # Absolute error at each pixel
    return nonconformity_scores

def get_pixelwise_prediction_interval(y_pred_new, nonconformity_scores, epsilon=0.05):
    
    n_calibration_samples = nonconformity_scores.shape[0]
    
    # Calculate the quantile of the nonconformity scores for each pixel
    masked_nonconformity_scores = ma.masked_equal(nonconformity_scores, 0)
    q = np.quantile(masked_nonconformity_scores, 1 - epsilon, axis=0)
    
    # Construct the prediction interval for each pixel
    lower_bound = y_pred_new - q
    upper_bound = y_pred_new + q
    
    return lower_bound, upper_bound


def get_pixelwise_prediction_interval_std(y_pred_new, nonconformity_scores, epsilon=0.05):
    
    n_calibration_samples = nonconformity_scores.shape[0]

    # Calculate the quantile of the nonconformity scores for each pixel
    masked_nonconformity_scores = ma.masked_equal(nonconformity_scores, 0)
    pixelwise_std = np.std(masked_nonconformity_scores, axis=0)  # Standard deviation at each pixel
    z_score = scipy.stats.norm.ppf(1 - epsilon / 2)
    # q = np.quantile(masked_nonconformity_scores, 1 - epsilon, axis=0)
    
    # Construct the prediction interval for each pixel
    lower_bound = y_pred_new - z_score * pixelwise_std
    upper_bound = y_pred_new + z_score * pixelwise_std
    
    return lower_bound, upper_bound


def plot_depth_with_uncertainty(vis_depth, lower_bound, upper_bound, gt_depth):
    # vis_depth[vis_depth < MIN_DEPTH] = MIN_DEPTH
    # vis_depth[vis_depth > MAX_DEPTH] = MAX_DEPTH
    dmin = np.percentile(vis_depth, 5)
    dmax = np.percentile(vis_depth, 95)
    normalizer = mpl.colors.Normalize(vmin=dmin, vmax=dmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    
    colormapped_im = (mapper.to_rgba(vis_depth)[:, :, :3] * 255).astype(np.uint8)
    depth_colored = cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2BGR)
    
    colormapped_im_lower = (mapper.to_rgba(lower_bound)[:, :, :3] * 255).astype(np.uint8)
    lower_bound_colored = cv2.cvtColor(colormapped_im_lower, cv2.COLOR_RGB2BGR)
    
    colormapped_im_upper = (mapper.to_rgba(upper_bound)[:, :, :3] * 255).astype(np.uint8)
    upper_bound_colored = cv2.cvtColor(colormapped_im_upper, cv2.COLOR_RGB2BGR)
    
    colormapped_im_gt = (mapper.to_rgba(gt_depth)[:, :, :3] * 255).astype(np.uint8)
    gt_colored = cv2.cvtColor(colormapped_im_gt, cv2.COLOR_RGB2BGR)
    
    cv2.namedWindow("depth", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    cv2.resizeWindow("depth", colormapped_im.shape[1], colormapped_im.shape[0])
    cv2.imshow("depth", depth_colored)
    
    cv2.namedWindow("lower_bound", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    cv2.resizeWindow("lower_bound", colormapped_im_lower.shape[1], colormapped_im_lower.shape[0])
    cv2.imshow("lower_bound", lower_bound_colored)
    
    cv2.namedWindow("upper_bound", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("upper_bound", colormapped_im_upper.shape[1], colormapped_im_upper.shape[0])
    cv2.imshow("upper_bound", upper_bound_colored)
    
    cv2.namedWindow("gt", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("gt", colormapped_im_gt.shape[1], colormapped_im_gt.shape[0])
    cv2.imshow("gt", gt_colored)
    # create a results directory and save the depth images in it in a folder with the sequence number and weights folder name
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # filenames are all the files in opt.data_path/pose
        
        sequence_id = int(opt.eval_split.split("_")[1])
        filenames = readlines(
            os.path.join(os.path.dirname(__file__), "splits", "mot",
                     "seq_{:02d}.txt".format(sequence_id)))
        
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
        pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")
        
        encoder_dict = torch.load(encoder_path)
        
        img_ext = '.png' if opt.png else '.jpg'
        dataset = datasets.KITTIMotDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           frame_idxs = [0, 1], num_scales=4, is_train=False, img_ext=img_ext)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
        pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
        pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
        
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))
        pose_decoder.load_state_dict(torch.load(pose_decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        pose_encoder.cuda()
        pose_encoder.eval()
        pose_decoder.cuda()
        pose_decoder.eval()

        pred_disps = []
        pred_poses = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        
        if opt.save_pred_disps:
            model_name = opt.load_weights_folder.split("/")[-1]
            depth_save_dir = os.path.join("results", "mot", model_name, 'depth', f"{sequence_id:04d}")
            os.makedirs(depth_save_dir, exist_ok=True)
            
        opt.frame_ids = [0, 1]  # pose network only takes two frames as input

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
                for i in opt.frame_ids:
                    data[("color_aug", i, 0)] = data[("color_aug", i, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                
                all_color_aug = torch.cat([data[("color_aug", i, 0)] for i in opt.frame_ids], 1)
                features = [pose_encoder(all_color_aug)]
                axisangle, translation = pose_decoder(features)

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                    
                if opt.save_pred_disps:
                    output_path = os.path.join(
                        depth_save_dir, f"{data['index'][0].item():06d}.npy")
                    np.save(output_path, cv2.resize(pred_disp[0], (1242, 375)))

                pred_disps.append(pred_disp)
                pred_poses.append(
                    transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
            
            # generate depth for the last frame
            input_color = data[("color", 1, 0)].cuda()
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            output = depth_decoder(encoder(input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            if opt.save_pred_disps:
                output_path = os.path.join(
                    depth_save_dir, f"{data['index'][0].item()+1:06d}.npy")
                np.save(output_path, cv2.resize(pred_disp[0], (1242, 375)))
            pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)
        pred_poses = np.concatenate(pred_poses)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]
            
    if opt.save_pred_poses:
        line_pred_poses = pred_poses.reshape(-1, 16)
        model_name = opt.load_weights_folder.split("/")[-1]
        save_path = os.path.join("results", 'mot', model_name, "pred_motion",  f"{sequence_id:04d}",)
        os.makedirs(save_path, exist_ok=True)
        np.savetxt(os.path.join(save_path, "pred_motion.txt"), line_pred_poses)

    # if opt.save_pred_disps:
    #     output_path = os.path.join(
    #         opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
    #     print("-> Saving predicted disparities to ", output_path)
    #     np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir,"mot", "gt_depths", f"seq_{sequence_id:02d}.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    # gt_poses_path = os.path.join(opt.data_path, "pose", "{:04d}".format(sequence_id), "pose.txt")
    # gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    # gt_global_poses = np.concatenate(
    #     (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    # gt_global_poses[:, 3, 3] = 1
    # gt_xyzs = gt_global_poses[:, :3, 3]
    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    nonconformity_scores = []
    # gt_local_poses = []

    for i in range(1, pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        # if opt.eval_split == "eigen" or "seq" in opt.eval_split:
        #     mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        #     crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
        #                      0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        #     crop_mask = np.zeros(mask.shape)
        #     crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        #     mask = np.logical_and(mask, crop_mask)

        # else:
        #     mask = gt_depth > 0

        # pred_depth = pred_depth[mask]
        # gt_depth = gt_depth[mask]

        # pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = 35
            pred_depth *= ratio

        # pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        # pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        
        nonconformity_scores.append(calculate_pixelwise_nonconformity_scores(pred_depth, gt_depth))
        
    nonconformity_scores = np.array(nonconformity_scores)

    lower_bound, upper_bound = get_pixelwise_prediction_interval_std(pred_depth, nonconformity_scores)
    if opt.vis_depth:
        plot_depth_with_uncertainty(pred_depth, lower_bound, upper_bound, gt_depth)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
