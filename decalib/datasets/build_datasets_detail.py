import os
import random
import cv2
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from glob import glob
import torch
def build_train(config, is_train=True):
    data_list = []
    data_list.append(SelfDataset(K=config.K, image_size=config.image_size, mediapipePath=config.mediapipePath))
    dataset = ConcatDataset(data_list)
    return dataset

class SelfDataset(Dataset):
    def __init__(self, K, image_size, mediapipePath='mediapipe_landmark_embedding.npz'):
        '''
        K must be less than 6
        '''
        self.mediapipe_idx = np.load(mediapipePath, allow_pickle=True, encoding='latin1')['landmark_indices'].astype(int)
        self.K = K
        self.image_size = image_size
        self.current_batch_index = {}
        # self.max_masks = 
        ################
        self.source_dirs = [f'/mnt/sdb/datasets/VGG-Face2_1st_labeld_pre{self.K}', f'/mnt/sdb/data>={self.K}']
        # self.source_dirs = ['/mnt/sdb/datasets/VGG-Face2_1st_labeld', f'/mnt/sdb/data>={self.K}']
        self.all_dirs = []

        # 모든 ID 디렉토리를 찾기
        for source_dir in self.source_dirs:
            self.all_dirs += glob(os.path.join(source_dir, '*', 'masks'))
            print(f"Found {len(self.all_dirs)} directories in {source_dir}")

        random.shuffle(self.all_dirs)
        self.current_batch_index = {dir_path: 0 for dir_path in self.all_dirs}  


        # self.dir_file_counts = {}
        # for dir_path in self.all_dirs:
        #     mask_files = glob(os.path.join(dir_path, '*.npy'))
        #     self.dir_file_counts[dir_path] = mask_files
        #     print(f"Directory {dir_path} contains {len(mask_files)} .npy files")
        ################

    def shuffle(self):
        random.shuffle(self.all_dirs)

    def __len__(self):
        return len(self.all_dirs)

    def __getitem__(self, idx):
        ################
        dir_path = self.all_dirs[idx]
        mask_files = glob(os.path.join(dir_path, '*.npy'))
        
        if len(mask_files) < self.K:
            # K개보다 적은 파일이 있는 디렉토리는 건너뜁니다.
            return self.__getitem__((idx + 1) % len(self.all_dirs))
        
        
        random.shuffle(mask_files)
        batches = [mask_files[i:i + self.K] for i in range(0, len(mask_files), self.K)]


        if len(batches[-1]) < self.K:
            batches.pop()

        batch_idx = self.current_batch_index[dir_path]
        selected_files = batches[batch_idx]

        self.current_batch_index[dir_path] = (batch_idx + 1) % len(batches)
        # selected_files = batches[0]  # 첫 번째 배치를 선택
        ################

        images_224_lists = []
        kpt_list = []
        dense_kpt_list = []
        mask_list = []

        for seg_path in selected_files:
            name = os.path.splitext(os.path.split(seg_path)[-1])[0]
            image_path = seg_path.replace('masks', 'images').replace('.npy', '.jpg')
            if not os.path.exists(image_path):
                image_path = seg_path.replace('masks', 'images').replace('.npy', '.png')
            kpt_path = seg_path.replace('masks', 'kpts')
            kpt_path_mp = seg_path.replace('masks', 'kpts_dense')

            try:
                lmks = np.load(kpt_path)
                dense_lmks = np.load(kpt_path_mp)
                image = imread(image_path) / 255.
                mask = self.load_mask(seg_path, image.shape[0], image.shape[1])
                mask = cv2.resize(mask, (224, 224))
            except Exception as e:
                print(f"Error loading data from {seg_path}: {e}")
                raise e

            images_224_lists.append(cv2.resize(image, (224, 224)).transpose(2, 0, 1))
            kpt_list.append(lmks)
            dense_kpt_list.append(dense_lmks[self.mediapipe_idx, :])
            mask_list.append(mask)
        # 중복 확인을 위한 로그 추가
        # print(f"Selected files: {selected_files}")
        # for img_path in selected_files:
        #     print(f"Image path: {img_path}")
        images_224_array = torch.from_numpy(np.array(images_224_lists)).type(dtype=torch.float32)
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)
        dense_kpt_array = torch.from_numpy(np.array(dense_kpt_list)).type(dtype=torch.float32)
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype=torch.float32)

        data_dict = {
            'image_224': images_224_array,
            'landmark': kpt_array,
            'landmark_dense': dense_kpt_array,
            'mask': mask_array
        }
        return data_dict

    def load_mask(self, maskpath, h, w):
    
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            mask = np.zeros((h, w))
            mask[vis_parsing_anno > 0] = 1.
            mask[vis_parsing_anno == 2] = 1.
            mask[vis_parsing_anno == 3] = 1.
            mask[vis_parsing_anno == 4] = 1.
            mask[vis_parsing_anno == 5] = 1.
            mask[vis_parsing_anno == 9] = 1.
            mask[vis_parsing_anno == 7] = 1.
            mask[vis_parsing_anno == 8] = 1.
            mask[vis_parsing_anno == 10] = 0  # hair
            mask[vis_parsing_anno == 11] = 0  # left ear
            mask[vis_parsing_anno == 12] = 0  # right ear
            mask[vis_parsing_anno == 13] = 0  # glasses
        else:
            mask = np.ones((h, w, 3))
        # except Exception as e:
        #     print(f"Error loading mask from {maskpath}: {e}")
        #     raise e
        return mask

# 예시로 사용할 config 객체를 정의합니다.
# class Config:
#     def __init__(self):
#         self.K = 5
#         self.image_size = 224
#         self.mediapipePath = '/home/cine/Documents/DJ/DECA/data/mediapipe_landmark_embedding.npz'
#         self.num_workers = 4  # DataLoader에서 사용할 워커 수
#         self.dataset = self

# config = Config()

# # 데이터셋을 빌드합니다.
# dataset = build_train(config)

# # DataLoader를 통해 데이터를 로드합니다.
# data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=config.num_workers)

# # 첫 번째 배치의 데이터를 가져옵니다.
# for batch in data_loader:
#     try:
#         print("image_224 shape:", batch['image_224'].shape)
#         print("landmark shape:", batch['landmark'].shape)
#         print("landmark_dense shape:", batch['landmark_dense'].shape)
#         print("mask shape:", batch['mask'].shape)
#         break
#     except Exception as e:
#         print(f"Error in DataLoader: {e}")
#         raise e