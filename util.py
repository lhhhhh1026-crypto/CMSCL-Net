import os
import numpy as np
import time
import math
import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_metric_learning import losses

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        labels = torch.Tensor(labels).to('cuda:0')
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        # print(logits)
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels)) /  2


class EpochBasedWeightedLoss(nn.Module):
    def __init__(self, total_epochs, initial_alpha=1.0):
        super(EpochBasedWeightedLoss, self).__init__()
        self.total_epochs = total_epochs
        self.initial_alpha = initial_alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.2)
        self.ntxent_loss = losses.NTXentLoss(temperature=0.2)

    def forward(self, logits, embeddings, labels, current_epoch):
        # 计算两种损失
        ce_loss = self.cross_entropy_loss(logits, labels)
        ntxent_loss = self.ntxent_loss(embeddings, labels)/4

        # 动态调整权重
        alpha = self.initial_alpha - (self.initial_alpha / self.total_epochs) * current_epoch
        alpha = max(0.1, alpha)  # 确保权重不低于最小值

        # 计算组合损失
        total_loss = alpha * ce_loss + (1 - alpha) * ntxent_loss
        return total_loss


def gaussian_kernel(kernel_size=9, sigma=1.0):
    # 生成一个核网格
    x = torch.arange(kernel_size).float()
    x = x - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(x, x)
    
    # 计算高斯函数
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def apply_lowpass_filter(images, kernel_size=9, sigma=2.0):
    # 获取图像批次大小
    
    images = torch.tensor(images, dtype=torch.float32)
    batch_size, channels, height, width = images.size()
    
    # 生成高斯核
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_size, kernel_size]
    
    # 扩展到多个通道
    kernel = kernel.repeat(channels, 1, 1, 1)  # [channels, 1, kernel_size, kernel_size]
    
    # 使用2D卷积进行滤波
    filtered_images = F.conv2d(images, kernel, padding=kernel_size // 2, groups=channels)
    filtered_images = filtered_images.view(-1,224,224)
    # print('filtered_images shape:',filtered_images.shape)
    return filtered_images

def max_filter_pytorch(img, kernel_size=3):
    # Ensure the image is a tensor and has the correct shape (batch, channels, height, width)
    img = img.view(-1,1,224,224)
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    
    # Apply max pooling (which acts as a max filter)
    # padding=kernel_size//2 ensures output size is the same as input size
    output_img = F.max_pool2d(img, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    # print('max_filter_pytorch shape:',output_img.shape)
    return output_img

def pc_from_txt(filename):
    """
    Loads a point cloud from a .txt file
    
    Parameters:
    filename:string - location for the .txt file
    
    Returns:
    point_cloud - numpy array of point cloud data, with points as rows (3 columns for 3d data)
    """
    point_cloud = np.loadtxt(filename)
    return point_cloud

def plot_downsample(point_cloud, sample_fraction = 0.1, random_seed=0, figsize=(10,10), is_scaled = True):
    """
    Plots a 3D point cloud using matplotlib
    
    Parameters:
    point_cloud - nx3 numpy array of the points to plot
    sample_fraction - downsampling fraction (float)
    random_seed - rng seed
    figsize - self explanatory (tuple)
    
    Returns:
    figure, axes for the plot
    """
    np.random.seed(random_seed)

    idx = np.random.randint(point_cloud.shape[0], size=int(point_cloud.shape[0]*sample_fraction))
    
    plot_points = point_cloud[idx,:]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    if is_scaled:
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.set_zlim((-1,1))
    
    ax.scatter(plot_points[:,0], plot_points[:,1], plot_points[:,2], marker='.')
    
    return fig, ax

def center_and_scale(point_cloud):
    """
    Recenters a point cloud to the origin and
    scales to [-1,1]^3
    
    Parameters:
    point_cloud - nx3 numpy array of points
    
    Returns
    scaled_cloud - centered and scaled point cloud
    """
    centered_cloud = point_cloud-np.mean(point_cloud, axis=0)
    return centered_cloud/np.max(abs(centered_cloud))

def project_point_np(point, camera_projection, f=1):
    """
    Perspective projection of a single point on to image plane, using numpy.
    
    Parameters:
    point - 3D point to be projected as numpy array, in real world coordinates - Dimension 4 NOT 1x4 
    camera_projection - 4x4 camera projection matrix as numpy array:
                        x_camera_tilde = camera_projection * x_world_tilde (with x as col. vectors)
                        camera_transform = r11 r12 r13 Tx
                                           r21 r22 r23 Ty
                                           r31 r32 r33 Tz 
                                           0   0   0   1
                                           
    f - distance from camera to image plane. Default=1

    Returns
    projected_point - projected coordinates in image plane - (real world coordinates - not pixels)
    depth - depth of projected point (from camera)
    
    _tilde denotes homegeneous coordinates
    """ 
    
    x_world_tilde = np.append(point, 1)
    
    x_camera_tilde = np.matmul(camera_projection, x_world_tilde)
    

    
    perspective_projection = np.array([
                            [f, 0, 0, 0],
                            [0, f, 0, 0],
                            [0, 0, 1, 0],
                            ])
    
    x_image_tilde = np.matmul(perspective_projection, x_camera_tilde)
    
    x_image = np.array([x_image_tilde[0]/x_image_tilde[2], x_image_tilde[1]/x_image_tilde[2]])
    
    return x_image, x_camera_tilde[2] #x,y coordinates in image plane, depth

def project_points_np(points, camera_projection, f=1):
    """
    Projects a list/numpy array of points in one go, using numpy.
    """
    projected_points_depths = [project_point_np(point, camera_projection, f) for point in points]
    projected_points = np.array([x[0] for x in projected_points_depths])
    projected_depths = np.array([x[1] for x in projected_points_depths])
    
    return projected_points, projected_depths

def project_points(cloud, camera_projection, f=1, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    colorlabel = False
    if cloud.shape[1] == 4:
        cloud = np.array(cloud).reshape(-1,4)
        cloud = torch.from_numpy(cloud).view(-1,4)

        color = cloud[:,-1].T.to(device=device)
        colorlabel = True
        cloud = cloud[:,0:3]
    elif cloud.shape[1] == 3:
        cloud = np.array(cloud).reshape(-1,3)
        cloud = torch.from_numpy(cloud).view(-1,3)
        
    x_world_tilde=torch.cat((cloud.clone().detach(), torch.ones(cloud.shape[0],1)), 1).transpose(0,1)  
    x_world_tilde=x_world_tilde.float()
    x_world_tilde=x_world_tilde.to(device=device)
    


    camera_projection = torch.tensor(camera_projection,
                                     dtype=torch.float32)
    
    perspective_projection = torch.tensor([
                            [f, 0, 0, 0],
                            [0, f, 0, 0],
                            [0, 0, 1, 0],
                            ], dtype=torch.float32)
    
    projection_matrix = torch.matmul(perspective_projection, camera_projection)
    projection_matrix = projection_matrix.to(device=device)
    
    x_image_tilde = torch.matmul(projection_matrix, x_world_tilde)
    if not colorlabel:
        color = x_image_tilde[2] /torch.max(x_image_tilde[2])
        
    return (x_image_tilde[0:2]/x_image_tilde[2]).transpose(0,1), color#x_image_tilde[2]


def depth_image_from_projection(points, depths, camera_fov_deg=90, image_dim=128, k=50, use_hard_min=False):
    """
    Plots a depth image from perspective projected points
    
    Parameters:
    points - nx2 numpy array of projected points
    depths - corresponding point depths from camera
    camera_fov_deg - camera FOV in degrees
    """
    
    #The maximum projected image coordinate based on camera FOV
    max_value = np.tan(np.radians(camera_fov_deg/2))
  
    no_points = len(points)
    points = points + max_value #Change range from -max->max to 0->2*max
    pixel_coords = torch.round(points/(2*max_value) * image_dim).long()

            
    if use_hard_min:
        max_coord = max(int(torch.max(pixel_coords)), image_dim)
        image = torch.zeros(size=(max_coord+1, max_coord+1))   

        for coord, depth in zip(pixel_coords, depths):
            i, j = int(coord[1]), int(coord[0])
            if (0 <= i < image_dim) and (0 <= j < image_dim): #If not outside the FOV
                if not(image[i, j]):
                    image[i, j] = depth
                elif depth < image[i, j]: #If there's something else there
                    image[i, j] = depth
                    
        return image
    
    else:
        pos_filter = torch.logical_and(pixel_coords[:,0]>=0, pixel_coords[:,1]>=0)
        max_filter = torch.logical_and(pixel_coords[:,0]<image_dim, pixel_coords[:,1]<image_dim)
        
        range_filter = torch.logical_and(pos_filter, max_filter)
        
        pixel_coords = pixel_coords[range_filter]
        depths = depths[range_filter]
        sparse = torch.sparse_coo_tensor(pixel_coords.t(), torch.pow(depths, -k), (image_dim, image_dim))
        sparse = sparse.coalesce()
        sparse = torch.pow(sparse, -1/k)
        return sparse.t().to_dense().to('cpu')   #image[0:image_dim, 0:image_dim]

    
    

def get_depth_images_from_cloud(points, camera_fov_deg=90, image_dim=128, f=1, camera_dist=1.4, k=50, use_hard_min=False):
    """
    Returns a set of 6 depth images from axis viewpoints, given the point cloud (Not the projected points)
    """
    camera_projections = [
    #Front to back
    np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, camera_dist],
        [0, 0, 0, 1]
    ]),

    #Back to front
    np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, camera_dist],
        [0, 0, 0, 1]
    ]),

    #Bottom up
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, camera_dist],
        [0, 0, 0, 1]
    ]),

    #Top down
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, camera_dist],
        [0, 0, 0, 1]
    ]),

    #Left to right
    np.array([
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, camera_dist],
        [0, 0, 0, 1]
    ]),

    #Right to left
    np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, camera_dist],
        [0, 0, 0, 1]
    ]),
    #  #45°
    # np.array([
    #     [np.cos(np.pi/4),-np.sin(np.pi/4), 0, 0          ],
    #     [0              , 0              , 1, 0          ],
    #     [np.sin(np.pi/4), np.cos(np.pi/4), 0, camera_dist],
    #     [0              , 0              , 0, 1          ]
    # ]),
    # #135°
    # np.array([
    #     [np.cos(3*np.pi/4),-np.sin(3*np.pi/4), 0, 0          ],
    #     [0                , 0                , 1, 0          ],
    #     [np.sin(3*np.pi/4), np.cos(3*np.pi/4), 0, camera_dist],
    #     [0              , 0                  , 0, 1          ]
    # ]),
    
    # #225°
    # np.array([
    #     [np.cos(5*np.pi/4),-np.sin(5*np.pi/4), 0, 0          ],
    #     [0                , 0                , 1, 0          ],
    #     [np.sin(5*np.pi/4), np.cos(5*np.pi/4), 0, camera_dist],
    #     [0              , 0                  , 0, 1          ]
    # ]),
    # #315°
    # np.array([
    #     [np.cos(7*np.pi/4),-np.sin(7*np.pi/4), 0, 0          ],
    #     [0                , 0                , 1, 0          ],
    #     [np.sin(7*np.pi/4), np.cos(7*np.pi/4), 0, camera_dist],
    #     [0              , 0                  , 0, 1          ]
    # ]),
    ]
    
    start = time.time()
    
    points_depths = [project_points(cloud=points, 
                                    camera_projection=camera_projection, 
                                    f=f) 
                     for camera_projection in camera_projections]
    
    end = time.time()
    #print("Time projecting: ", end-start)
    
    start=time.time()

    depth_images = [depth_image_from_projection(points=points, 
                                                depths=depths, 
                                                camera_fov_deg=camera_fov_deg, 
                                                image_dim=image_dim, 
                                                k=k,
                                                use_hard_min=use_hard_min).unsqueeze(axis=0) 
                    for points, depths in points_depths]
    
    depth_images = torch.cat(depth_images)
    end=time.time()
    #print("Time imaging: ", end-start)
    
    
    
    return depth_images

def plot_depth_image(depth_image):
    """
    Plots a single depth image using the matplotlib OO interface
    """
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(depth_image, origin='lower')
    
    return fig, ax
    
def plot_depth_images(depth_images, nrows=2, figsize=None, cmap='viridis'):
    """
    Plots a full set of projections (6 as in the paper) across 2 rows
    using the matplotlib OO interface
    """
    from_dataset = False
    if len(depth_images.shape) == 4:
        from_dataset = True
    
    no_images = len(depth_images)
    
    ncols = math.ceil(no_images/nrows)
    
    if figsize == None:
        figsize = (5*ncols, 5*nrows)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    
    for i in range(no_images):
        if from_dataset:
            if nrows>1:
                ax[i//ncols,i%ncols].imshow(depth_images[i][0], origin='lower', cmap = cmap)
            else:
                ax[i].imshow(depth_images[i][0], origin='lower', cmap = cmap)
        else:
            if nrows>1:
                ax[i//ncols,i%ncols].imshow(depth_images[i], origin='lower', cmap = cmap)
            else:
                ax[i].imshow(depth_images[i], origin='lower', cmap = cmap)
        
    return fig, ax    


