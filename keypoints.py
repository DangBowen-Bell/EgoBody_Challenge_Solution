import os
import os.path as osp
import torch
import pandas as pd
import pickle
import numpy as np
import json
import glob
import ast
import PIL.Image as pil_img
from PIL import ImageDraw

import config
from models import SMPL
from utils.geometry import transform_global_orient
from utils.egobody_utils import *


data_root = config.EGOBODY_ROOT
dbw_root = config.DBW_ROOT
model_path = config.OTHER_DATA_ROOT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

period = 'train'
save_viz = False
use_full = True

data_info_df = pd.read_csv(os.path.join(data_root, 'data_info_release.csv'))
recording_name_list = list(data_info_df['recording_name'])

start_frame_list = list(data_info_df['start_frame'])
end_frame_list = list(data_info_df['end_frame'])
body_idx_fpv_list = list(data_info_df['body_idx_fpv'])
start_frame_dict = dict(zip(recording_name_list, start_frame_list))
end_frame_dict = dict(zip(recording_name_list, end_frame_list))
body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))

data_split_df = pd.read_csv(os.path.join(data_root, 'data_splits.csv'))
recording_name_list = list(data_split_df[period])
while np.nan in recording_name_list:
    recording_name_list.remove(np.nan)

total_num = 0
pv_num = 0
valid_num = 0
time_num = 0
true_valid_num = 0

gt_keypoints_2d = []
new_global_orients = []
for recording_name in recording_name_list:
    print('Processing ', recording_name)
    fitting_dir = osp.join(data_root, 'smpl_interactee', recording_name)
    
    body_idx = int(body_idx_fpv_dict[recording_name].split(' ')[0])
    body_gender = body_idx_fpv_dict[recording_name].split(' ')[1]
    
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)

    pv_dir = glob.glob(os.path.join(data_root, 'egocentric_color', recording_name, '202*'))[0]
    
    pv_path_list = glob.glob(os.path.join(pv_dir, 'PV', '*_frame_*.jpg'))
    pv_path_list = sorted(pv_path_list)
    
    frame_id_list = [osp.basename(x).split('.')[0].split('_', 1)[1] for x in pv_path_list]
    frame_id_dict = dict(zip(frame_id_list, pv_path_list))

    op_keypoints_path = osp.join(pv_dir, 'keypoints.npz')
    op_keypoints = np.load(op_keypoints_path)
    valid_frame_id_list = [osp.basename(x).split('.')[0].split('_', 1)[1] for x in op_keypoints['imgname']]

    valid_frame_path = osp.join(pv_dir, 'valid_frame.npz')
    valid_frame = np.load(valid_frame_path)
    valid_frame_dict = dict(zip(valid_frame_id_list, valid_frame['valid']))

    timestamp_list = [osp.basename(x).split('_')[0] for x in pv_path_list]
    timestamp_dict = dict(zip(timestamp_list, frame_id_list))
    pv_info_path = glob.glob(os.path.join(pv_dir, '*_pv.txt'))[0]
    with open(pv_info_path) as f:
        lines = f.readlines()
    cx, cy, w, h = ast.literal_eval(lines[0])
    fx_dict = {}
    fy_dict = {}
    holo_pv2world_trans_dict = {}
    for i, frame in enumerate(lines[1:]):
        frame = frame.split((','))
        cur_timestamp = frame[0]  
        cur_fx = float(frame[1])
        cur_fy = float(frame[2])
        cur_pv2world_transform = np.array(frame[3:20]).astype(float).reshape((4, 4))
        if cur_timestamp in timestamp_dict.keys():
            cur_frame_id = timestamp_dict[cur_timestamp]
            fx_dict[cur_frame_id] = cur_fx
            fy_dict[cur_frame_id] = cur_fy
            holo_pv2world_trans_dict[cur_frame_id] = cur_pv2world_transform

    calib_trans_dir = osp.join(data_root, 'calibrations', recording_name) 
    holo2kinect_dir = osp.join(calib_trans_dir, 'cal_trans', 'holo_to_kinect12.json')
    with open(holo2kinect_dir, 'r') as f:
        trans_holo2kinect = np.array(json.load(f)['trans'])
    trans_kinect2holo = np.linalg.inv(trans_holo2kinect)

    total_num += end_frame_dict[recording_name] - start_frame_dict[recording_name] + 1
    pv_num += len(frame_id_list)
    valid_num += len(valid_frame_id_list)
    time_num += len(fx_dict)
    for k in valid_frame_dict.keys():
        if valid_frame_dict[k]:
            true_valid_num += 1

    for frame in range(start_frame_dict[recording_name], end_frame_dict[recording_name]+1):
        frame_id = 'frame_{}'.format("%05d" % frame)

        if not (frame_id in frame_id_list and \
                frame_id in valid_frame_id_list):
            continue

        if not use_full and \
           not (valid_frame_dict[frame_id] and frame_id in fx_dict.keys()) and \
           period != 'test':
                continue

        fitting_path = osp.join(fitting_dir, 'body_idx_{}'.format(body_idx), 'results', frame_id, '000.pkl')
        with open(fitting_path, 'rb') as f:
            param = pickle.load(f)
        torch_param = {}
        torch_param['transl'] = torch.tensor(param['transl']).to(device)
        torch_param['global_orient'] = torch.tensor(param['global_orient']).to(device)
        torch_param['betas'] = torch.tensor(param['betas']).to(device)
        torch_param['body_pose'] = torch.tensor(param['body_pose']).to(device)

        # output = smpl(return_verts=True, **torch_param)
        # joints = output.joints.detach().cpu().numpy().squeeze()
        output = smpl(betas=torch_param['betas'], 
                      body_pose=torch_param['body_pose'], 
                      global_orient=torch_param['global_orient'])
        # world -> RGB (kinect)
        joints = output.joints.detach().cpu().numpy().squeeze() + param['transl']

        cur_fx = fx_dict[frame_id]
        cur_fy = fy_dict[frame_id]
        cur_pv2world_transform = holo_pv2world_trans_dict[frame_id]
        cur_world2pv_transform = np.linalg.inv(cur_pv2world_transform)

        add_trans = np.array([[1.0, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]]) 
        
        # joints = points_coord_trans(joints, trans_kinect2holo)  
        # joints = points_coord_trans(joints, cur_world2pv_transform)
        # joints = points_coord_trans(joints, add_trans)

        # RGB (kinect) -> world (holo) -> -> RGB (holo)
        trans_mtx = add_trans.dot(cur_world2pv_transform.dot(trans_kinect2holo))
        joints = points_coord_trans(joints, trans_mtx)
        # trans_mtxs.append(trans_mtx) 

        global_orient = np.squeeze(param['global_orient'].copy())
        transl = np.squeeze(param['transl'].copy())
        new_global_orient, new_transl = transform_global_orient(global_orient, transl, trans_mtx)
        new_global_orient = new_global_orient.astype(np.float32)
        new_global_orients.append(new_global_orient)

        # output = smpl(betas=torch_param['betas'], 
        #               body_pose=torch_param['body_pose'], 
        #               global_orient=torch.tensor([new_global_aa]).to(device))
        # joints = output.joints.detach().cpu().numpy().squeeze() + new_transl

        camera_center_holo = torch.tensor([cx, cy]).view(-1, 2)
        camera_holo_kp = create_camera(camera_type='persp_holo',
                                       focal_length_x=torch.tensor([cur_fx]).to(device).unsqueeze(0),
                                       focal_length_y=torch.tensor([cur_fy]).to(device).unsqueeze(0),
                                       center=camera_center_holo,
                                       batch_size=1).to(device=device)

        joints = torch.from_numpy(joints).float().to(device).unsqueeze(0) 
        gt_joints_2d = camera_holo_kp(joints)
        gt_joints_2d = gt_joints_2d.squeeze().detach().cpu().numpy()
        # print(gt_joints_2d)
        
        gt_keypoint_conf = np.ones((len(gt_joints_2d),1))
        for j in range(len(gt_joints_2d)):
            if gt_joints_2d[j][0] < 0 or \
               gt_joints_2d[j][0] >= 1920 or \
               gt_joints_2d[j][1] < 0 or \
               gt_joints_2d[j][1] >= 1080:
                gt_keypoint_conf[j][0] = 0.0
        gt_keypoint_2d = np.concatenate((gt_joints_2d, gt_keypoint_conf), axis=1)
        gt_keypoints_2d.append(gt_keypoint_2d)

        if save_viz:
            img_path = os.path.join(pv_dir, 'PV', frame_id_dict[frame_id])
            img = cv2.imread(img_path)[:, :, ::-1]
            output_img = pil_img.fromarray((img).astype(np.uint8))
            draw = ImageDraw.Draw(output_img)
            # draw.ellipse((100, 200,
            #               120, 220), 
            #               fill=(255, 0, 0, 0))
            for k in range(len(gt_joints_2d)):
                if k>=24:
                    draw.ellipse((gt_joints_2d[k][0] - 4, gt_joints_2d[k][1] - 4,
                            gt_joints_2d[k][0] + 4, gt_joints_2d[k][1] + 4), 
                            fill=(255, 0, 0, 0))
                else:
                    draw.ellipse((gt_joints_2d[k][0] - 4, gt_joints_2d[k][1] - 4,
                                gt_joints_2d[k][0] + 4, gt_joints_2d[k][1] + 4), 
                                fill=(0, 255, 0, 0))
            output_img.convert('RGB')
            output_img = output_img.resize((int(w), int(h)))
            output_img.save('./output/' + frame_id + '_output.jpg')
            break

dir_name = 'full_gt_keypoints' if use_full else 'gt_keypoints'

gt_keypoints_2d = np.array(gt_keypoints_2d)
gt_keypoints_2d_path = osp.join(dbw_root, dir_name, period)
if not save_viz:
    np.save(gt_keypoints_2d_path, gt_keypoints_2d)
print('Saved to ', gt_keypoints_2d_path, '.npy')
print('Shape: ', gt_keypoints_2d.shape)

new_global_orients = np.array(new_global_orients)
new_global_orients_path = osp.join(dbw_root, dir_name, period + '_global_orient')
if not save_viz:
    np.save(new_global_orients_path, new_global_orients)
print('Saved to ', new_global_orients_path, '.npy')
print('Shape: ', new_global_orients.shape)
