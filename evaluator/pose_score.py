from transformation import *
import numpy as np
import os
import argparse
from evaluator_base import per_frame_scale_alignment
from tartanair_evaluator import TartanAirEvaluator
from transformation import pose_quats2motion_ses, motion_ses2pose_quats, eulers2qauts
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='score_pose')

parser.add_argument('--gt_path',
                    type=str,
                    help='path to the root of the KITTI mot data',
                    required=True)
parser.add_argument('--pred_path',
                    type=str,
                    help='which split to export gt from',
                    required=True,)

# parser.add_argument('--seq',
#                     type=str,
#                     help='which seq to export gt from',
#                     required=False,)
                    # choices=["eigen", "eigen_benchmark"])
opt = parser.parse_args()

def pose_score(opt):
    gtposes = np.loadtxt(opt.gt_path) # N x 12
    gtposes_ses = SEs2ses(gtposes) # N x 6
    gt_motions_SE = pose2motion(gtposes) # N-1 x 12
    gt_motions = SEs2ses(gt_motions_SE) # N-1 x 6
    gt_poses_quat = SEs2quats(gtposes) # N x 7
    
    pred_motions_SE = np.loadtxt(opt.pred_path) # N-1 x 16
    pred_motions_SE = pred_motions_SE[:,:12] # N-1 x 12
    pred_motions = SEs2ses(pred_motions_SE) # N-1 x 6
    
    
    estmotion_scale = per_frame_scale_alignment(gt_motions, pred_motions)
    estposes = motion_ses2pose_quats(estmotion_scale)
    evaluator = TartanAirEvaluator()
    results = evaluator.evaluate_one_trajectory(gt_poses_quat, estposes, scale=False, kittitype=True)
    
    print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

    # plt.figure()
    # plt.plot(gtposes[:,3], gtposes[:,7], linestyle='dashed',c='k')
    
    # save results and visualization
    plot_traj(results['gt_aligned'], results['est_aligned'], vis=True, title='ATE %.4f' %(results['ate_score']))
    # np.savetxt('results/'+testname+'.txt',results['est_aligned'])


    
def plot_traj(gtposes, estposes, vis=False, savefigname=None, title=''):
    fig = plt.figure(figsize=(4,4))
    # cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    plt.plot(gtposes[:,0],gtposes[:,1], linestyle='dashed',c='k')
    plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth','TartanVO'])
    plt.title(title)

    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(gtposes[:,0],gtposes[:,1],gtposes[:,2], c='k')
    # ax.plot(estposes[:, 0], estposes[:, 1], estposes[:, 2],c='#ff7f0e')
    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_zlabel('z (m)')
    # ax.legend(['Ground Truth','TartanVO'])
    # ax.set_title(title)
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)    
    
if __name__ == '__main__':
    pose_score(opt)
    # pose_score(opt)
    # export_gt_depths_kitti_mot(opt)