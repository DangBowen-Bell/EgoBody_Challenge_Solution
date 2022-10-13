import os.path as osp
import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
import cv2
import pickle
import numpy as np
import ast
from torchvision.transforms import Normalize

import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa, cutout, motion_blur


def load_egoset_dataset(args, period):
    data_info_df = pd.read_csv(osp.join(args.data_root, 'data_info_release.csv'))
    recording_name_list = list(data_info_df['recording_name'])

    start_frame_list = list(data_info_df['start_frame'])
    end_frame_list = list(data_info_df['end_frame'])
    body_idx_fpv_list = list(data_info_df['body_idx_fpv'])
    start_frame_dict = dict(zip(recording_name_list, start_frame_list))
    end_frame_dict = dict(zip(recording_name_list, end_frame_list))
    body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))

    data_split_df = pd.read_csv(osp.join(args.data_root, 'data_splits.csv'))

    data = {
        'img_paths': [],
        'centers': [],
        'scales': [],
        'keypoints': [],
        'op_keypoints': [],
        # 'trans': [],
        'fitting_paths': [],
        'genders': [],
        'global_orient': []
    }

    recording_name_list = list(data_split_df[period])
    while np.nan in recording_name_list:
        recording_name_list.remove(np.nan)

    dir_name = 'full_gt_keypoints' if args.use_full else 'gt_keypoints'
    gt_keypoints_path = osp.join(args.dbw_root, dir_name, period + '.npy')
    gt_keypoints = np.load(gt_keypoints_path)
    # trans_path = osp.join(args.data_root, dir_name, period + '_trans.npy')
    # trans = np.load(trans_path)
    global_orient_path = osp.join(args.dbw_root, dir_name, period + '_global_orient.npy')
    global_orient = np.load(global_orient_path)

    i = 0
    for recording_name in recording_name_list:
        pv_dir = glob.glob(osp.join(args.data_root, 'egocentric_color', recording_name, '202*'))[0]

        pv_path_list = glob.glob(osp.join(pv_dir, 'PV', '*_frame_*.jpg'))
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
        pv_info_path = glob.glob(osp.join(pv_dir, '*_pv.txt'))[0]
        with open(pv_info_path) as f:
            lines = f.readlines()
        cx, cy, w, h = ast.literal_eval(lines[0])
        fx_dict = {}
        fy_dict = {}
        holo_pv2world_trans_dict = {}
        for _, frame in enumerate(lines[1:]):
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

        center_list = op_keypoints['center']
        scale_list = op_keypoints['scale']
        op_keypoints_list = op_keypoints['keypoints']
        center_dict = dict(zip(valid_frame_id_list, center_list))
        scale_dict = dict(zip(valid_frame_id_list, scale_list))
        op_keypoints_dict = dict(zip(valid_frame_id_list, op_keypoints_list))

        fitting_dir = osp.join(args.data_root, 'smpl_interactee', recording_name)

        body_idx = int(body_idx_fpv_dict[recording_name].split(' ')[0])
        body_gender = body_idx_fpv_dict[recording_name].split(' ')[1]
        gender_dict = {'female': -1, 'neutral': 0, 'male': 1}
        body_gender = gender_dict[body_gender]

        for frame in range(start_frame_dict[recording_name], end_frame_dict[recording_name]+1):
            frame_id = 'frame_{}'.format("%05d" % frame)
            
            if not (frame_id in frame_id_list and \
                    frame_id in valid_frame_id_list):
                continue

            if not args.use_full and \
            not (valid_frame_dict[frame_id] and frame_id in fx_dict.keys()) and \
            period != 'test':
                continue
            
            img_path = osp.join(pv_dir, 'PV', frame_id_dict[frame_id])
            
            center = center_dict[frame_id]
            scale = scale_dict[frame_id]

            op_keypoint = op_keypoints_dict[frame_id]
            gt_keypoint_pad = np.zeros((24, 3))
            op_keypoint = np.concatenate([op_keypoint, gt_keypoint_pad])

            # tran = trans[i]
            keypoint = gt_keypoints[i]
            new_global_orient = global_orient[i]
            i += 1

            fitting_path = osp.join(fitting_dir, 'body_idx_{}'.format(body_idx), 'results', frame_id, '000.pkl')
            
            data['img_paths'].append(img_path)
            data['centers'].append(center)
            data['scales'].append(scale)
            data['keypoints'].append(keypoint)
            data['op_keypoints'].append(op_keypoint)
            # data['trans'].append(tran)
            data['fitting_paths'].append(fitting_path)
            data['genders'].append(body_gender)
            data['global_orient'].append(new_global_orient)
        
    return data


class EgoSetDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset='egoset', use_augmentation=False, is_train=True, period='train'):
        super(EgoSetDataset, self).__init__()
        self.options = options   
        self.dataset = dataset     
        self.use_augmentation = use_augmentation
        self.is_train = is_train
        self.period = period

        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        
        data = load_egoset_dataset(options, period)
        self.img_paths = data['img_paths']
        self.centers = data['centers']
        self.scales = data['scales']
        self.keypoints = data['keypoints']
        self.op_keypoints = data['op_keypoints']
        # self.trans = data['trans']
        self.fitting_paths = data['fitting_paths']
        self.genders = data['genders']
        self.global_orient = data['global_orient']
        self.length = len(self.img_paths)

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)     # per channel pixel-noise
        rot = 0             # rotation
        sc = 1              # scaling
        co = 0              # cutout
        blur = -1           # blur
        blur_size = 1       # blur kernel size
        ox = 0              # offset along x axis
        oy = 0              # offset along y axis
        if self.is_train and self.use_augmentation:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 
                                   1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                      max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                     max(1-self.options.scale_factor, 1+np.random.randn()*self.options.scale_factor))

            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

            if self.options.cutout_factor > 0:
                co = np.random.uniform(0, self.options.cutout_factor)

            if self.options.use_blur and np.random.uniform() <= 0.1:
                blur = np.random.randint(4)
                blur_size = np.random.randint(len(constants.BLUR_SIZES))
                blur_size = constants.BLUR_SIZES[blur_size]

            if self.options.use_offset and np.random.uniform() <= 0.4:
                ox = np.random.randint(-6, 6)
                oy = np.random.randint(-3, 3)
        
        return flip, pn, rot, sc, co, blur, blur_size, ox, oy

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, blur, blur_size):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # blur the image
        if blur != -1:
            rgb_img = motion_blur(rgb_img, blur, blur_size)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224), float, [0,1]
        # rgb_img = np.transpose(rgb_img.astype('float32'), (2,0,1)) / 255.0 
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1] / constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scales[index].copy()
        center = self.centers[index].copy()

        # Get augmentation parameters
        # flip, pn, rot, sc = self.augm_params()
        flip, pn, rot, sc, co, blur, blur_size, ox, oy = self.augm_params()
        # flip = 0           
        # pn = np.ones(3)     
        # rot = 0             
        # sc = 1
        # co = 0
        # blur = -1
        # blur_size = 1
        # ox = 0
        # oy = 0

        center[0] += ox
        center[1] += oy
        
        # Load image
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)[:,:,::-1].copy().astype(np.float32)
        orig_shape = np.array(img.shape)[:2]

        # Get SMPL parameters, if available
        global_orient = self.global_orient[index]
        fitting_path = self.fitting_paths[index]
        with open(fitting_path, 'rb') as f:
            param = pickle.load(f)
            pose = np.concatenate((global_orient, 
                                   np.squeeze(param['body_pose'].copy())))
            betas = np.squeeze(param['betas'].copy())

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn, blur, blur_size)

        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = img_path

        item['pose_3d'] = torch.zeros(24, 4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        if self.options.use_op:
            keypoint = self.op_keypoints[index].copy()
        else:          
            keypoint = self.keypoints[index].copy()
        keypoint = self.j2d_processing(keypoint, center, sc*scale, rot, flip)

        if self.options.cutout_factor > 0:
            co = constants.IMG_RES * co
            img, keypoint = cutout(img, keypoint, co)

        # (3,224,224), float, [0,1]
        img = np.transpose(img.astype('float32'), (2,0,1)) / 255.0
        img = torch.from_numpy(img).float()
        item['img'] = self.normalize_img(img)

        # convert to normalized coordinates
        item['keypoints'] = torch.from_numpy(keypoint).float()
        
        # trans = self.trans[index].copy()
        # item['trans'] =  torch.from_numpy(trans).float()

        item['has_smpl'] = 1
        item['has_pose_3d'] = 0
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.genders[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        item['maskname'] = ''
        item['partname'] = ''

        return item

    def __len__(self):
        return self.length
