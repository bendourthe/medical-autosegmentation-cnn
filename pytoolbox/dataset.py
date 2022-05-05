####################
# LIBRARIES IMPORT #
####################

import os
import math
import numpy as np
from scipy import ndimage
from dicompylercore import dicomparser
import imageio
import cv2

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

####################
# CLASS DEFINITION #
####################

#----------------------------------------------------------------------------------------------------#
class ProcessedDataset(Dataset):
    
    def __init__(self,
                 train_dicoms_path,
                 train_labels_path,
                 input_shape=256,
                 augment_params={'aug_prob': 0.5,
                                 'rotation': False,
                                 'rot_range': 5,
                                 'warping': False,
                                 'warp_range': 0.8,
                                 'cropping': False,
                                 'crop_range': [0.5, 0.8]}):
        '''
		DOURTHE TECHNOLOGIES - CONFIDENTIAL
	    Unpublished Copyright © 2022 Dourthe Technologies (dourthetechnologies.com)

	    Created on: Mon Apr 11 2022 
	    Author: Benjamin Dourthe (benjamin@dourthe.tech)

	    Description:
	    ------------

            Class enabling the definition of a dataset with on-the-fly data augmentation
            for the training of a Multiscale Pyramid 2D Convolutional Neural Network aiming at
            performing the automated segmentation of Magnetic Resonance Images.

        Parameters:
        -----------

            train_dicoms_path: full path towards DICOM images.
                (type: string)
            train_labels_path: full path towards labels.
                (type: string)
            input_shape: specifies shape of input image (e.g. input_shape=128 for input images of size 128x128).
                (type: integer) (default=256)
            augment_params: dictionary containing all augmentation parameters.
                (type: dictionary)
                details:
                    aug_prob: defines augmentation probability (i.e., percentage of chance that each iteration will be augmented).
                        (type: float) (default=0.5)
                    rotation: specifies whether rotation augmentation will be applied.
                        (type: boolean - True to apply rotation augmentation, False NOT to apply rotation augmentation) (default=False)
                    rot_range: define rotation angle range (in degrees).
                        (e.g. if set to 5, each transformed image is randomly rotated by an angle comprised between -5 and +5 degrees)
                        (type: integer) (default=5)
                    warping: specifies whether warping augmentation will be applied.
                        (type: boolean - True to apply warping augmentation, False NOT to apply warping augmentation) (default=False)
                    warp_range: define warping range.
                        (e.g. if set to 0.8, each transformed image is randomly resized along the x- and y-axes by a factor
                        comprised between 80% and 100% of the original image shape with different factors for each direction.
                        The resize image is then re-resize to the original image shape by filling the cropped pixels with black pixels,
                        therefore giving the impression of warping)
                        (type: float) (default=0.8)
                    cropping: specifies whether cropping augmentation will be applied.
                        (type: boolean - True to apply cropping augmentation, False NOT to apply cropping augmentation) (default=False)
                    crop_range: define cropping range.
                        (e.g. if crop_range = [0.5, 0.8], each transformed image will be randomly cropped along the x- and y-axes
                        by a factor comprised between 50% and 80% of the original image dimensions)
                        (type: list of floats) (default=[0.5, 0.8])

        Returns:
        --------

			img_tensor: torch.FloatTensor containing batches of training images
			lab_tensor: torch.FloatTensor containing batches of training labels

        '''

        #############################
        # PARAMETERS INITIALIZATION #
        ############################# 
        
        # Define paths towards DICOMs and labels directories
        self.train_dicoms_path = train_dicoms_path
        self.train_labels_path = train_labels_path

        # Define dimension of the input images (only one value because images are or will be turned to square)
        self.input_shape = input_shape
            
        # Define augmentation parameters
        self.augment_params = augment_params
        self.augment_params['crop_ratio'] = np.random.uniform(self.augment_params['crop_range'][0],
                                                              self.augment_params['crop_range'][1])        
        
        #########################
        # DICOMS PRE-PROCESSING #
        #########################
        
        # Define list of scans
        self.dicom_files_list = sorted(os.listdir(train_dicoms_path))

        # Initialize empty list that will contain all training images
        # (array of shape = (number of samples, x-axis resolution, y-axis resolution)
        self.img_list = []
        
        # Loop through every scan folder
        for file in self.dicom_files_list:
            
            # Read DICOM
            dicom = dicomparser.DicomParser(os.path.abspath(os.path.join(self.train_dicoms_path, file).replace('\\', '/')))

            # Convert DICOM file to numpy array 
            arr = np.array(dicom.GetImage())

            # Convert array to selected data type
            arr = arr.astype(np.float32) 

            # Normalize pixel values between 0 and 1
            arr = arr + np.abs(arr.min())
            arr = arr/(arr.max())
            
            # Resize image to input shape (to ensure consistency)
            arr = cv2.resize(arr, dsize=(self.input_shape, self.input_shape), interpolation=cv2.INTER_CUBIC)

            # Append current converted DICOM to list
            self.img_list.append(arr)
        
        # Convert list of images to array with selected data type
        self.images = np.array(self.img_list).astype(np.float32)
        
        #########################
        # LABELS PRE-PROCESSING #
        #########################
        
        # Define list of scans
        self.labels_files_list = sorted(os.listdir(train_labels_path))
        
        # Initialize empty list that will contain all training labels
        # (array of shape = (number of samples, x-axis resolution, y-axis resolution)
        self.lab_list = []
        
        # Loop through every scan folder
        for file in self.labels_files_list:

            # Read label
            if file.endswith('.npy'):
                lab = np.load(os.path.join(self.train_labels_path, file))
            elif file.endswith('.jpg') or file.endswith('.png'):
                lab = imageio.imread(os.path.join(self.train_labels_path, file))
            if len(np.shape(lab)) > 2:
                lab = lab[:, :, 0]

            # Resize label to input shape (to ensure consistency)
            lab = cv2.resize(lab, dsize=(self.input_shape, self.input_shape), interpolation=cv2.INTER_CUBIC)
            
            # Append current label to list
            self.lab_list.append(lab)
        
        # Convert list of labels to array with selected data type
        self.labels = np.array(self.lab_list).astype(np.float32)
        
    def __getitem__(self, index):

        # Read images and labels
        img_arr = self.images[index]
        lab_arr = self.labels[index]
            
        # AUGMENTATION #1: RANDOM ROTATION
        if self.augment_params['rotation']:
            if np.random.uniform(0, 1) >= self.augment_params['aug_prob']:
                # Calculate random angle
                angle = np.random.uniform(-self.augment_params['rot_range'], self.augment_params['rot_range'])
                # Rotate images and labels
                img_arr = ndimage.rotate(img_arr, angle, axes=(0, 1), reshape=False, order=1, cval=0)
                lab_arr = ndimage.rotate(lab_arr, angle, axes=(0, 1), reshape=False, order=1, cval=0)

        # AUGMENTATION #2: RANDOM WARPINGH
        if self.augment_params['warping']:
            if np.random.uniform(0, 1) >= self.augment_params['aug_prob']:
                # Save original images and labels shapes
                x_shape, y_shape = img_arr.shape[0], img_arr.shape[1]
                # Generate random warping factors along x- and y-axes
                warp_x = np.random.uniform(self.augment_params['warp_range'], 1)
                warp_y = np.random.uniform(self.augment_params['warp_range'], 1)
                # Warp images and labels
                img_arr = ndimage.zoom(img_arr, (warp_x, warp_y), order=1, cval=0)
                lab_arr = ndimage.zoom(lab_arr, (warp_x, warp_y), order=1, cval=0)
                # Calculate number of rows and columns for padding
                n_rows = x_shape - img_arr.shape[0]
                n_cols = y_shape - img_arr.shape[1]
                # Calculate padding shape
                if n_rows%2 == 0 and n_cols%2 == 0:
                    pad_shape = ((int(n_rows/2), int(n_rows/2)),
                                 (int(n_cols/2), int(n_cols/2)))
                elif n_rows%2 != 0 and n_cols%2 == 0:
                    pad_shape = ((math.ceil(n_rows/2), math.floor(n_rows/2)),
                                 (int(n_cols/2), int(n_cols/2)))
                elif n_rows%2 == 0 and n_cols%2 != 0:
                    pad_shape = ((int(n_rows/2), int(n_rows/2)),
                                 (math.ceil(n_cols/2), math.floor(n_cols/2)))
                elif n_rows%2 != 0 and n_cols%2 != 0:
                    pad_shape = ((math.ceil(n_rows/2), math.floor(n_rows/2)),
                                 (math.ceil(n_cols/2), math.floor(n_cols/2)))
                # Pad images and labels to match original shape
                img_arr = np.pad(img_arr, pad_shape, 'constant', constant_values=0)
                lab_arr = np.pad(lab_arr, pad_shape, 'constant', constant_values=0)

        # AUGMENTATION #3: RANDOM CROPPING
        if self.augment_params['cropping']:
            if np.random.uniform(0, 1) >= self.augment_params['aug_prob']:
                # Save original images and labels shapes
                x_shape, y_shape = img_arr.shape[0], img_arr.shape[1]
                # Calculate cropping ratio
                # Define cropped image size
                crop_size = int(x_shape*self.augment_params['crop_ratio'])
                # Define random coordinates for bottom left corner of the cropping area
                x1 = int(np.random.uniform(0, x_shape-1 - crop_size))
                y1 = int(np.random.uniform(0, y_shape-1 - crop_size))
                # Define random coordinates for top right corner of the cropping area
                x2 = x1 + crop_size
                y2 = y1 + crop_size
                # Crop images and labels
                img_arr = img_arr[x1:x2+1, y1:y2+1]
                lab_arr = lab_arr[x1:x2+1, y1:y2+1]
                # Upsample cropped images and labels to match with original image size
                with torch.no_grad():
                    img_arr = torch.FloatTensor(img_arr).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
                    img_arr = F.upsample(img_arr,
                                         (1, x_shape, y_shape),
                                         mode='trilinear').squeeze().squeeze().detach().numpy()
                    lab_arr = torch.FloatTensor(lab_arr).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
                    lab_arr = F.upsample(lab_arr,
                                         (1, x_shape, y_shape),
                                         mode='trilinear').squeeze().squeeze().detach().numpy()
                    
        # Convert images and labels from numpy.ndarray to torch.FloatTensor format
        img_tensor = torch.FloatTensor(img_arr).unsqueeze(0)
        lab_tensor = torch.FloatTensor(lab_arr)

        return img_tensor, lab_tensor
        
    def __len__(self):      
        
        return len(self.images)