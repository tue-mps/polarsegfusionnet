from torch.utils.data import Dataset
import numpy as np
import os
from torchvision.transforms import Resize,CenterCrop
import torchvision.transforms as transform
import pandas as pd
from PIL import Image

class RADIal(Dataset):

    def __init__(self, root_dir,statistics=None,difficult=False):

        self.root_dir = root_dir
        self.statistics = statistics
        # sampleid.csv file contains the object detection labels.
        # we use this file just to read the frame names as we don't perform object detection.
        # This file is named as "labels.csv" in the original dataset.
        self.sampleid = pd.read_csv(os.path.join(root_dir,'sampleid.csv')).to_numpy()
       
        # Keeps only easy samples
        if(difficult==False):
            ids_filters=[]
            ids = np.where( self.sampleid[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.sampleid = self.sampleid[ids_filters]

        # Gather each input entries by their sample id
        self.unique_ids = np.unique(self.sampleid[:,0])
        self.label_dict = {}
        for i,ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.sampleid[:,0]==ids)[0]
            self.label_dict[ids]=sample_ids
        self.sample_keys = list(self.label_dict.keys())

        self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512,448))

    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, index):
        
        # Get the sample id
        sample_id = self.sample_keys[index]

        # Read the Radar FFT data
        radar_name = os.path.join(self.root_dir,'radar_FFT',"fft_{:06d}.npy".format(sample_id))
        input = np.load(radar_name,allow_pickle=True)
        radar_FFT = np.concatenate([input.real,input.imag],axis=2)
        if(self.statistics is not None):
            for i in range(len(self.statistics['input_mean'])):
                radar_FFT[...,i] -= self.statistics['input_mean'][i]
                radar_FFT[...,i] /= self.statistics['input_std'][i]

        # Read the segmentation map
        segmap_name = os.path.join(self.root_dir,'radar_Freespace',"freespace_{:06d}.png".format(sample_id))
        segmap = Image.open(segmap_name) # [512,900]
        # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap = self.crop(segmap)
        # and we resize to half of its size
        segmap = np.asarray(self.resize(segmap))==255

        # Read the BEV image
        bev_img_name = os.path.join(self.root_dir, 'BEV_Polar_Python_resized', "image_{:06d}.jpg".format(sample_id))
        bev_image = np.asarray(Image.open(bev_img_name))

        # Read the camera image
        cam_img_name = os.path.join(self.root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
        cameraimage = np.asarray(Image.open(cam_img_name))

        return radar_FFT, segmap, bev_image, cameraimage, cam_img_name
