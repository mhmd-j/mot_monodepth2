from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map_MOT, generate_depth_map_odom

def gen_mot_txt_files():
    data_path = '/home/apera/mhmd/kittiMOT/data_kittiMOT/training'
    seq_list = os.listdir(os.path.join(data_path, "image_02"))

    save_dir = os.path.join(os.path.dirname(__file__), "splits", "mot")
    os.makedirs(os.path.join(save_dir, ), exist_ok=True)

    for seq in seq_list:
        img_file = os.path.join(data_path, "image_02", seq)
        img_list = sorted(os.listdir(img_file))
        img_list.sort()
        # pop the last one because we need two consecutive images
        img_list.pop()
        # delete the zeros in the beginning of the seq string
        with open(os.path.join(save_dir, f'seq_{int(seq):02d}.txt'), 'w') as f:
            for img in img_list:
                _img_ind = int(img.split('.')[0])
                f.write(f"{seq} {_img_ind} l"+ '\n')
                

def export_gt_depths_kitti_mot():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI mot data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,)
    
    parser.add_argument('--seq',
                        type=str,
                        help='which seq to export gt from',
                        required=False,)
                        # choices=["eigen", "eigen_benchmark"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    if opt.seq is not None:
        seq_no = [f"seq_{int(opt.seq.split('_')[1]):02d}.txt"]
    else:
        seq_no = sorted(os.listdir(split_folder))[2:]
        
    os.makedirs(os.path.join(split_folder, "gt_depths"), exist_ok=True)
    for seq in seq_no:
        lines = readlines(os.path.join(split_folder, seq))

        print("Exporting ground truth depths for {}, {}".format(opt.split, seq))

        gt_depths = []
        for line in lines:

            folder, frame_id, _ = line.split()
            frame_id = int(frame_id)
            if opt.split == "mot":
                calib_dir = os.path.join(opt.data_path, "calib", folder.split("/")[0] + ".txt")
                # velo_filename = os.path.join(opt.data_path, folder,
                #                             "velodyne_points/data", "{:010d}.bin".format(frame_id))
                velo_filename = os.path.join(
                    opt.data_path,
                    "velodyne",
                    folder,
                    "{:06d}.bin".format(int(frame_id)),
                    )
                gt_depth = generate_depth_map_MOT(calib_dir, velo_filename, im_shape=(375, 1242), cam=2, vel_depth=True)

            gt_depths.append(gt_depth.astype(np.float32))
        
        output_path = os.path.join(split_folder, "gt_depths", seq.split('.')[0] + ".npz")

        print("Saving to {}".format(opt.split))

        np.savez_compressed(output_path, data=np.array(gt_depths))


def export_gt_depths_kitti_odom():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI mot data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,)
    
    parser.add_argument('--seq',
                        type=str,
                        help='which seq to export gt from',
                        required=False,)
                        # choices=["eigen", "eigen_benchmark"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    if opt.seq is not None:
        seq_names = [f"{opt.seq}.txt"]
    else:
        seq_names = sorted(os.listdir(split_folder))
        
    os.makedirs(os.path.join(split_folder, "gt_depths"), exist_ok=True)
    for seq in seq_names:
        lines = readlines(os.path.join(split_folder, seq))

        print("Exporting ground truth depths for {}, {}".format(opt.split, seq))

        gt_depths = []
        for line in lines:

            folder, frame_id, _ = line.split()
            folder = int(folder)
            frame_id = int(frame_id)
            if opt.split == "odom":
                calib_dir = os.path.join(opt.data_path, 'sequences', f"{folder:02d}", "calib.txt")
                velo_filename = os.path.join(
                    opt.data_path,
                    "sequences",
                    f"{folder:02d}",
                    "velodyne",
                    "{:06d}.bin".format(int(frame_id)),
                    )
                gt_depth = generate_depth_map_odom(calib_dir, velo_filename, im_shape=(375, 1242), cam=2, vel_depth=True)

            gt_depths.append(gt_depth.astype(np.float32))
        
        output_path = os.path.join(split_folder, "gt_depths", seq.split('.')[0] + ".npz")

        print("Saving to {}".format(opt.split))

        np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    # export_gt_depths_kitti_mot()
    gen_mot_txt_files()
