import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import util

class ScannetDatasetWholeScene(Dataset):
    """
    可被 DataLoader 使用的 Scannet 数据集。
    支持点云采样和可选的数据增强。
    """

    def __init__(self, root, metadata_file, aug=False, transforms=[], seed=None):
        super().__init__()
        self.root = root
        self.aug = aug
        self.transforms = transforms
        self.seed = seed

        # 🔧 新增：统一随机数生成器
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        # 初始化点云和标签列表
        self.scene_points_list = []
        self.semantic_labels_list = []

        self.image_dim = 224

        # 获取文件列表
        self.file_list = os.listdir(root)
        self.filelist = [os.path.splitext(f)[0] for f in self.file_list]

        # 读取元数据
        meta_frame = pd.read_csv(metadata_file, keep_default_na=False)
        species_label = []
        for file in self.filelist:
            species_label.append(meta_frame.loc[meta_frame['id'] == file, 'sp'].values[0])

        # 去掉不需要的类别
        keep_filelist, keep_species = [], []
        for f, l in zip(self.filelist, species_label):
            if l not in ['NA', 'JUNIPE', 'QUERCUS', 'DEAD']:
                keep_filelist.append(f)
                keep_species.append(l)
        self.filelist = keep_filelist
        species_label = keep_species

        # 标签编码
        species_num = []
        for sp in species_label:
            if sp == 'PINNIG':
                species_num.append(0)
            elif sp == 'PINPIN':
                species_num.append(1)
            elif sp == 'PINSYL':
                species_num.append(2)
            elif sp == 'QUEFAG':
                species_num.append(3)
            elif sp == 'QUEILE':
                species_num.append(4)

        self.point_label = np.column_stack((self.filelist, species_num))

        # 读取点云
        for i, file in enumerate(self.filelist):
            points = np.loadtxt(os.path.join(root, self.point_label[i, 0] + '.txt'))
            points = points[:, :3]
            points = points - np.mean(points, axis=0)
            points = points / np.max(abs(points))
            self.scene_points_list.append(points)
            self.semantic_labels_list.append(int(self.point_label[i, 1]))

    def get_depth_image(self, points):
        return util.get_depth_images_from_cloud(points=points, image_dim=self.image_dim)

    # 🔧 所有随机操作用 self.rng
    def random_rotation(self, point_cloud, dgcnn_point, min_rotation=0, max_rotation=2*np.pi):
        theta = np.random.rand() * (max_rotation - min_rotation) + min_rotation
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        point_cloud = point_cloud.dot(Rz.T)
        dgcnn_point = dgcnn_point.dot(Rz.T)
        return point_cloud, dgcnn_point

    def random_translation(self, point_cloud, dgcnn_point, min_translation=0., max_translation=0.1):
        sign = np.sign(np.random.rand(1) - 0.5)
        tran = np.random.rand(3) * (max_translation - min_translation) + min_translation
        point_cloud = point_cloud + sign * tran
        dgcnn_point = dgcnn_point + sign * tran
        return point_cloud, dgcnn_point

    def random_scaling(self, point_cloud, dgcnn_point, min_scale=0.5, max_scale=1.5):
        scale = np.random.rand(1) * (max_scale - min_scale) + min_scale
        point_cloud = point_cloud * scale
        dgcnn_point = dgcnn_point * scale
        return point_cloud, dgcnn_point

    def dataaug(self, points, dgcnn_points):
        if 'rotation' in self.transforms:
            points, dgcnn_points = self.random_rotation(points, dgcnn_points)
        if 'translation' in self.transforms:
            points, dgcnn_points = self.random_translation(points, dgcnn_points)
        if 'scaling' in self.transforms:
            points, dgcnn_points = self.random_scaling(points, dgcnn_points)
        # points_list = points
        # dgcnn_points_list = dgcnn_points
        return points, dgcnn_points

    def trad_feature(self, batch_data):
        layer_interval = 0.02
        layers = np.arange(-1., 1., layer_interval)
        # batch_result = []

        
        canopy_widths = []
        for h in layers:
            layer_points = batch_data[(batch_data[:, 2] >= h) & (batch_data[:, 2] < h + layer_interval)]
            if len(layer_points) < 3:
                canopy_widths.append(0)
                continue
            layer_points_xy = layer_points[:, :2]
            x_min, y_min = np.min(layer_points_xy, axis=0)
            x_max, y_max = np.max(layer_points_xy, axis=0)
            canopy_widths.append((x_max - x_min + y_max - y_min) / 2)
        # batch_result.append(canopy_widths)
        return np.array(canopy_widths)

    # 🔧 改写 __getitem__，一次只返回一个样本
    def __getitem__(self, index):
        points = self.scene_points_list[index]
        label = self.semantic_labels_list[index]

        # 采样
        if self.aug:
            if points.shape[0] < 2048:
                point_idx = np.random.choice(points.shape[0], 2048, replace=True)
            else:
                point_idx = np.random.choice(points.shape[0], 2048, replace=False)
        else:
            if points.shape[0] < 2048:
                point_idx = self.rng.choice(points.shape[0], 2048, replace=True)
            else:
                point_idx = self.rng.choice(points.shape[0], 2048, replace=False)

        dgcnn_points = points[point_idx, :]

        batch_data = points
        batch_dgcnn_data = dgcnn_points

        if self.aug:
            batch_data, batch_dgcnn_data = self.dataaug(batch_data, batch_dgcnn_data)

        batch_dgcnn_data = np.stack(batch_dgcnn_data)
        trad_feat = self.trad_feature(batch_dgcnn_data)

        # depth_images = torch.zeros((1, 6, self.image_dim, self.image_dim))

        depth_images = self.get_depth_image(batch_data)
        
        # print(self.get_depth_image(batch_data).shape)

        return trad_feat, depth_images, batch_dgcnn_data, label

    def __len__(self):
        return len(self.scene_points_list)

import torch
import numpy as np
from torch.utils.data import DataLoader

def test_random_samples(root, metadata_file, num_samples=5, aug=True, transforms=[]):
    dataset = ScannetDatasetWholeScene(
        root=root,
        metadata_file=metadata_file,
        aug=aug,
        transforms=transforms
    )
    print(f"数据集总样本数: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0  # Windows 下测试用 0
    )
    
    for i, (trad_feat, depth_img, points, label) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  trad_feat: {trad_feat.shape}")
        print(f"  depth_img: {depth_img.shape}")
        print(f"  points: {points.shape}")
        print(f"  label: {label.shape}")

    print("\n✅ 随机样本输出维度验证通过")

if __name__ == "__main__":
    root = r"D:\\Anaconda3\\envs\\TreeClassfication\\full_points_model\\data\\train"
    metadata_file = r"D:\\Anaconda3\\envs\\TreeClassfication\\full_points_model\\data\\meta\\tree-meta.csv"

    test_random_samples(root=root, metadata_file=metadata_file)



# import os
# import random
# import numpy as np
# import torch
# import pandas as pd
# from tqdm import tqdm
# from torch.utils.data import Dataset
# import util
# from scipy.spatial import ConvexHull, distance
# import pickle
# import hashlib


# def _fixed_rng(seed_source):
#     """根据样本唯一标识生成固定随机数生成器"""
#     if isinstance(seed_source, int):
#         seed = seed_source
#     else:  # 比如文件路径 / 名称
#         seed = int(hashlib.md5(str(seed_source).encode()).hexdigest(), 16) % (2**32)
#     return np.random.RandomState(seed)


# class ScannetDatasetWholeScene():
#     # 预测数据集制作----给每一个点来进行预测值
#     def __init__(self, root, metadata_file, aug=False, balanced=False, transforms=[], seed=None):

#         self.root = root
#         self.balanced = balanced
#         self.transforms = transforms
#         self.aug = aug
#         self.seed = seed

#         # 🔧 新增：统一随机数生成器
#         self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

#         # 整个场景的点云
#         self.scene_points_list = []
#         # 整个场景的标签列表
#         self.semantic_labels_list = []

#         self.image_dim = 224
#         self.camera_fov_deg = 90
#         self.f = 1
#         self.camera_dist = 1.4

#         self.file_list = os.listdir(root)
#         print('file_list', len(self.file_list))
#         # ----------------------------------------------------------------------------------------------
#         self.filelist = [os.path.splitext(file)[0] for file in self.file_list]
#         # 读取元数据
#         meta_frame = pd.read_csv(metadata_file, keep_default_na=False)
#         # 按照点云文件的读取顺序将所有点云的种类读取进来
#         species_label = []
#         for file in self.filelist:
#             species_label.append(meta_frame.loc[meta_frame['id'] == file, 'sp'].values[0])

#         # 去除掉不要的点云文件及标签
#         for i, label in enumerate(species_label):
#             if label in ['NA', 'JUNIPE', 'QUERCUS', 'DEAD']:
#                 species_label.pop(i)
#                 self.filelist.pop(i)

#         species_num = []
#         for i, specie in enumerate(species_label):
#             if species_label[i] == 'PINNIG':
#                 species_num.append(0)
#             elif species_label[i] == 'PINPIN':
#                 species_num.append(1)
#             elif species_label[i] == 'PINSYL':
#                 species_num.append(2)
#             elif species_label[i] == 'QUEFAG':
#                 species_num.append(3)
#             elif species_label[i] == 'QUEILE':
#                 species_num.append(4)
#         self.point_label = np.column_stack((self.filelist, species_num))

#         print(self.point_label.shape)

#         for i, file in enumerate(self.filelist):
#             points = np.loadtxt(root + self.point_label[i, 0] + '.txt')
#             points = points[:, :3]  # 读取到整个点云的数据
#             points = points - np.mean(points, axis=0)
#             points = points / np.max(abs(points))
#             self.scene_points_list.append(points)
#             self.semantic_labels_list.append(self.point_label[i, 1])

#         self.semantic_labels_list = [int(x) for x in self.semantic_labels_list]
#         self.index_list = list(range(len(self.scene_points_list)))

#     def get_depth_image(self, points):
#         return util.get_depth_images_from_cloud(points=points,
#                                                 image_dim=self.image_dim,
#                                                 )

#     # 🔧 修改：使用 self.rng 代替 np.random
#     def random_rotation(self, point_cloud, dgcnn_point, min_rotation=0, max_rotation=2*np.pi):
#         theta = self.rng.rand() * (max_rotation - min_rotation) + min_rotation

#         Rz = np.array([
#             [np.cos(theta), -np.sin(theta), 0],
#             [np.sin(theta), np.cos(theta), 0],
#             [0, 0, 1]
#         ])
#         point_cloud = np.dot(point_cloud, Rz.T)
#         dgcnn_point = np.dot(dgcnn_point, Rz.T)
#         return point_cloud, dgcnn_point

#     def random_translation(self, point_cloud, dgcnn_point, min_translation=0., max_translation=0.1):
#         sign = np.sign(self.rng.rand(1) - 0.5)
#         tran = self.rng.rand(3) * (max_translation - min_translation) + min_translation
#         point_cloud = point_cloud + sign * tran
#         dgcnn_point = dgcnn_point + sign * tran
#         return point_cloud, dgcnn_point

#     def random_scaling(self, point_cloud, dgcnn_point, min_scale=0.5, max_scale=1.5):
#         scale = self.rng.rand(1) * (max_scale - min_scale) + min_scale
#         point_cloud = scale * point_cloud
#         dgcnn_point = scale * dgcnn_point
#         return point_cloud, dgcnn_point

#     def dataaug(self, points, dgcnn_points):
#         for i, (point, dgcnn_point) in enumerate(zip(points, dgcnn_points)):
#             if 'rotation' in self.transforms:
#                 point, dgcnn_point = self.random_rotation(point, dgcnn_point)

#             if 'translation' in self.transforms:
#                 point, dgcnn_point = self.random_translation(point, dgcnn_point)

#             if 'scaling' in self.transforms:
#                 point, dgcnn_point = self.random_scaling(point, dgcnn_point)
#             points[i] = point
#             dgcnn_points[i] = dgcnn_point
#         return points, dgcnn_points

#     def trad_feature(self, batch_data):
#         layer_interval = 0.02
#         layers = np.arange(-1., 1., layer_interval)

#         batch_result = []

#         for points in batch_data:
#             canopy_widths = []

#             for h in layers:
#                 layer_points = points[(points[:, 2] >= h) & (points[:, 2] < h + layer_interval)]

#                 if len(layer_points) < 3:
#                     canopy_widths.append(0)
#                     continue

#                 layer_points_xy = layer_points[:, :2]

#                 x_min, y_min = np.min(layer_points_xy, axis=0)
#                 x_max, y_max = np.max(layer_points_xy, axis=0)

#                 length = x_max - x_min  # x方向的范围
#                 width = y_max - y_min   # y方向的范围
#                 canopy_width = (length + width) / 2
#                 canopy_widths.append(canopy_width)

#             batch_result.append(canopy_widths)

#         return np.array(batch_result)

#     def __getitem__(self, index):

#         datalen = list(range(len(self.index_list)))
#         batchlen = datalen[index]
#         indexs = self.index_list[index]
#         batch_data = []
#         batch_dgcnn_data = []
#         batch_label = []

#         for index in indexs:
#             points = self.scene_points_list[index]
#             label = self.semantic_labels_list[index]

#             # 🔧 修改：采样部分也使用 self.rng 控制
#             if self.aug :
#                 if points.shape[0] < 2048:
#                     point_idx = self.rng.choice(points.shape[0], 2048, replace=True)
#                 else:
#                     point_idx = self.rng.choice(points.shape[0], 2048, replace=False)
#             else:
#                 if points.shape[0] < 2048:
#                     point_idx = self.rng.choice(points.shape[0], 2048, replace=True)
#                 else:
#                     point_idx = self.rng.choice(points.shape[0], 2048, replace=False)

#             dgcnn_points = points[point_idx, :]

#             batch_data.append(points)
#             batch_dgcnn_data.append(dgcnn_points)
#             batch_label.append(label)

#         batch_data_aug = batch_data

#         if self.aug:
#             batch_data_aug, batch_dgcnn_data = self.dataaug(batch_data, batch_dgcnn_data)

#         batch_dgcnn_data = np.stack(batch_dgcnn_data)
#         trad_feat = self.trad_feature(batch_dgcnn_data)

#         batchsize = len(batchlen)
#         depth_images = torch.zeros(size=(batchsize, 6, self.image_dim, self.image_dim))

#         for i in range(len(batch_data_aug)):
#             depth_images[i] = self.get_depth_image(batch_data_aug[i])
#         batch_label = torch.tensor(batch_label).to('cuda:0')
#         return batch_data, trad_feat, depth_images, batch_dgcnn_data, batch_label

#     def __len__(self):
#         return len(self.scene_points_list)

#     def shuffle_indices(self):
#         print('shuffle_indices')
#         random.shuffle(self.index_list)   # 🔒 保持不改动（如果要复现可改成 self.rng.shuffle）






# # import os
# # import random
# # import numpy as np
# # import torch
# # import pandas as pd
# # from tqdm import tqdm
# # from torch.utils.data import Dataset
# # import util
# # from scipy.spatial import ConvexHull,distance
# # import pickle
# # # torch.manual_seed(0)
# # # torch.cuda.manual_seed(0)
# # # torch.cuda.manual_seed_all(0)
# # # torch.random.manual_seed(0)
# # # np.random.seed(0)
# # # random.seed(0)
# # # from clip.feature_map import extract_feature_maps

# # import hashlib

# # def _fixed_rng(seed_source):
# #     """根据样本唯一标识生成固定随机数生成器"""
# #     if isinstance(seed_source, int):
# #         seed = seed_source
# #     else:  # 比如文件路径 / 名称
# #         seed = int(hashlib.md5(str(seed_source).encode()).hexdigest(), 16) % (2**32)
# #     return np.random.RandomState(seed)



# # class ScannetDatasetWholeScene():
# #     #预测数据集制作----给每一个点来进行预测值
# #     # prepare to give prediction on each points
# #     def __init__(self, root,metadata_file,aug = False, balanced=False, transforms=[], seed=None):
        
# #         self.root = root
# #         self.balanced = balanced
# #         self.transforms = transforms
# #         self.aug = aug
# #         self.seed = seed

# #         # 整个场景的点云
# #         self.scene_points_list = [] 
# #         # self.dgcnn_points_list = []
# #         # self.seg_pred_list = [] 
# #         # self.dataaug_points_list = []
# #         # 整个场景的标签列表       
# #         self.semantic_labels_list = []

# #         self.image_dim = 224
# #         self.camera_fov_deg = 90
# #         self.f = 1
# #         self.camera_dist = 1.4

# #         self.file_list = os.listdir(root)
# #         # self.index_list = list(range(len(self.file_list)))  # 初始化索引列表
# #         print('file_list',len(self.file_list))
# # #----------------------------------------------------------------------------------------------
# #         self.filelist = [os.path.splitext(file)[0] for file in self.file_list]
# #         #读取元数据
# #         meta_frame = pd.read_csv(metadata_file, keep_default_na=False)
# #         #按照点云文件的读取顺序将所有点云的种类读取进来
# #         species_label = []
# #         for file in self.filelist:
# #             species_label.append(meta_frame.loc[meta_frame['id'] == file,'sp'].values[0])

# #         #去除掉不要的点云文件及标签
# #         for i,label in enumerate(species_label):
# #             if label in ['NA','JUNIPE','QUERCUS','DEAD']:
# #                 species_label.pop(i)
# #                 self.filelist.pop(i)
        
# #         species_num = []
# #         for i,specie in enumerate(species_label):
# #             if species_label[i] == 'PINNIG':
# #                 species_num.append(0)
# #             elif species_label[i] == 'PINPIN':
# #                 species_num.append(1)
# #             elif species_label[i] == 'PINSYL':
# #                 species_num.append(2)
# #             elif species_label[i] == 'QUEFAG':
# #                 species_num.append(3)
# #             elif species_label[i] == 'QUEILE':
# #                 species_num.append(4)
# #         self.point_label = np.column_stack((self.filelist, species_num))
        
# #         print(self.point_label.shape)

# #         for i,file in enumerate(self.filelist):
# #             # print('file',file)
# #             # print('point_label file',self.point_label[i,0])
# #             points = np.loadtxt(root + self.point_label[i,0] + '.txt')
# #             # seg_pred = np.loadtxt(seg_root + self.point_label[i,0] + '.txt').reshape(-1,1)
# #             points = points[:, :3] #读取到整个点云的数据
# #             points = points-np.mean(points,axis=0)
# #             points = points/np.max(abs(points))  

# #             # if points.shape[0]<2048:
# #             #     numpoints = points.shape[0]
# #             #     point_idx = np.random.choice(numpoints, 2048, replace=True)
# #             #     dgcnn_points = points[point_idx,:]
# #             # else:
# #             #     numpoints = points.shape[0]
# #             #     point_idx = np.random.choice(numpoints, 2048, replace=False)
# #             #     dgcnn_points = points[point_idx,:]
# #             self.scene_points_list.append(points)
# #             # self.dgcnn_points_list.append(dgcnn_points)
# #             # self.seg_pred_list.append(seg_pred)
# #             self.semantic_labels_list.append(self.point_label[i,1])

            
# #         # self.dataaug_points_list = self.scene_points_list
# #         # assert len(self.scene_points_list) == len(self.semantic_labels_list) == len(self.seg_pred_list)
# #         self.semantic_labels_list = [int(x) for x in self.semantic_labels_list]

        
# #         self.index_list = list(range(len(self.scene_points_list)))  # 初始化索引列表

        
        
            

# #     def get_depth_image(self, points):
# #         return util.get_depth_images_from_cloud(points=points, 
# #                                                  image_dim=self.image_dim, 
# #                                                  )
                                    


# #     def random_rotation(self, point_cloud,dgcnn_point, min_rotation=0, max_rotation=2*np.pi):
# #         theta = np.random.rand() * (max_rotation - min_rotation) + min_rotation
        
# #         Rz = np.array([
# #             [np.cos(theta), -np.sin(theta), 0],
# #             [np.sin(theta),  np.cos(theta), 0],
# #             [0            ,              0, 1]
# #         ])
# #         point_cloud = np.dot(point_cloud, Rz.T)
# #         dgcnn_point = np.dot(dgcnn_point, Rz.T)
# #         return point_cloud, dgcnn_point

# #     def random_translation(self, point_cloud, dgcnn_point, min_translation=0., max_translation=0.1):
# #         sign = np.sign(np.random.rand(1) - 0.5)
# #         tran = np.random.rand(3)*(max_translation - min_translation) + min_translation
# #         point_cloud = point_cloud +  sign * tran
# #         dgcnn_point = dgcnn_point +  sign * tran
# #         return point_cloud, dgcnn_point

# #     def random_scaling(self, point_cloud, dgcnn_point, min_scale=0.5, max_scale=1.5):
# #         scale = np.random.rand(1)*(max_scale - min_scale) + min_scale
# #         point_cloud = scale * point_cloud
# #         dgcnn_point = scale * dgcnn_point
# #         return point_cloud, dgcnn_point

# #     def dataaug(self,points,dgcnn_points):
        
# #         for i, (point, dgcnn_point) in enumerate(zip(points, dgcnn_points)):
# #             if 'rotation' in self.transforms:
# #                 point, dgcnn_point = self.random_rotation(point, dgcnn_point)

# #             if 'translation' in self.transforms:
# #                 point, dgcnn_point = self.random_translation(point, dgcnn_point)

# #             if 'scaling' in self.transforms:
# #                 point, dgcnn_point = self.random_scaling(point, dgcnn_point)
# #             points[i] = point 
# #             dgcnn_points[i] = dgcnn_point
# #         return points, dgcnn_points
    
# #     def trad_feature(self, batch_data):
# #         layer_interval = 0.02
# #         layers = np.arange(-1., 1., layer_interval)
        
# #         batch_result = []
        
# #         for points in batch_data:
# #             # 对每个点云去重一次即可
# #             # points = np.unique(points, axis=0)
# #             canopy_widths = []
            
# #             for h in layers:
# #                 # 获取当前层范围内的点
# #                 layer_points = points[(points[:, 2] >= h) & (points[:, 2] < h + layer_interval)]
                
# #                 # 如果该层点数不足，跳过
# #                 if len(layer_points) < 3:
# #                     canopy_widths.append(0)
# #                     continue
                
# #                 # 投影到 XY 平面
# #                 layer_points_xy = layer_points[:, :2]
                
# #                 x_min, y_min = np.min(layer_points_xy, axis=0)
# #                 x_max, y_max = np.max(layer_points_xy, axis=0)
                
# #                 # 计算长度和宽度
# #                 length = x_max - x_min  # x方向的范围
# #                 width = y_max - y_min   # y方向的范围
# #                 canopy_width = (length+width)/2
# #                 canopy_widths.append(canopy_width)
            
# #             batch_result.append(canopy_widths)

# #         # 将结果转为 numpy 数组以便后续处理
# #         return np.array(batch_result)

# #     def __getitem__(self, index):
        
    
# #         datalen = list(range(len(self.index_list)))
# #         batchlen = datalen[index]
# #         indexs = self.index_list[index]
# #         batch_data = []
# #         batch_dgcnn_data = []
# #         # seg_pred_batch = []
# #         batch_label = []
# #         for index in indexs:
# #             points = self.scene_points_list[index]
# #             # dgcnn_data = self.dgcnn_points_list[index]
# #             # seg_pred = self.seg_pred_list[index] 
# #             label = self.semantic_labels_list[index]
            
# #             if self.aug:  # 训练模式，保持随机采样
# #                 if points.shape[0] < 2048:
# #                     point_idx = np.random.choice(points.shape[0], 2048, replace=True)
# #                 else:
# #                     point_idx = np.random.choice(points.shape[0], 2048, replace=False)
# #             else:  # 测试模式，随机采样但受 seed 控制
# #                 rng = np.random.RandomState(self.seed + index)  # 全局 seed 控制 + index 保证不同样本不一样
# #                 if points.shape[0] < 2048:
# #                     point_idx = rng.choice(points.shape[0], 2048, replace=True)
# #                 else:
# #                     point_idx = rng.choice(points.shape[0], 2048, replace=False)

# #             dgcnn_points = points[point_idx, :]

# #             batch_data.append(points)
# #             batch_dgcnn_data.append(dgcnn_points)
# #             batch_label.append(label)
# #         batch_data_aug = batch_data
        
# #         if self.aug == True:
# #             batch_data_aug, batch_dgcnn_data = self.dataaug(batch_data, batch_dgcnn_data)
# #         batch_dgcnn_data = np.stack(batch_dgcnn_data)
# #         trad_feat = self.trad_feature(batch_dgcnn_data)

# #         # pointcloud_seg = np.concatenate((batch_dgcnn_data,seg_pred_batch),axis = 2)
# #         batchsize = len(batchlen)

# #         depth_images = torch.zeros(size=(batchsize, 6, self.image_dim, self.image_dim))

# #         for i in range(len(batch_data_aug)):
# #             depth_images[i] = self.get_depth_image(batch_data_aug[i])
# #         batch_label = torch.tensor(batch_label).to('cuda:0')
# #         return batch_data, trad_feat,depth_images, batch_dgcnn_data, batch_label
       

# #     def __len__(self):
# #         return len(self.scene_points_list)
    
# #     def shuffle_indices(self):
# #         print('shuffle_indices')
# #         random.shuffle(self.index_list)
   
# #         # if self.transforms is not None:
# #         #     print('data aug')
# #         #     self.dataaug()


    






























































