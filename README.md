# Automated Segmentation of Medical Images Using a Convolutional Neural Network


## Objectives
To train a Neural Network to automatically segment a selection of regions of interest (ROIs) from medical imaging data.


## Neural Network Architecture
The selected architecture is a Multiscale Pyramid 2D Convolutional Neural Network (Dourthe et al. (2021) [1]), which was chosen based on its reported ability to accurately extract contextual and morphological information from medical images at various scales.


## How to Use

### Requirements
Create a new Python 3.10 environment and install requirements.txt within this environment.
You will also need to run the following commands to install PyTorch and torchvision:
- conda install pytorch -c pytorch
- pip install torchvision

### Data Management

#### Data Structure
In order to allow the code to run successfully, it is recommended to organize the training data using the following structure:
<pre>
project_directory
└─ data
   └── train
       ├── dicoms
       │   └── contains all the raw images available in the training dataset (each scan axial slice represents a sample)
       └── labels
           └── contains all the segmentation files for the corresponding images available in the training dataset</pre>

#### Data Format
Here are the supported data formats:
- dicoms: DICOM format (.dcm)
- labels: multiple formats supported: .npy, .jpg, .png

#### Data Labelling
The images and corresponding segmentation files should have the same filenames. Otherwise the code will not be able to run successfully, as it will be looking for matching pairs of images and labels based on filenames during training.
For example, if a DICOM image is named 'slice_1.dcm', the corresponding segmentation file should be named 'slice_1.npy' or 'slice_1.jpg' or 'slice_1.png'.

#### Data Segmentation
Segmentation files can be generated using the software of choice (3D Slicer, ImageJ, Photoshop, etc.), as long as:
- The resulting files are in one of the supported formats (.npy, .jpg or .png)
- The filenames match with the corresponding segmented images
- One segmentation file is generated per slice

In addition, the pixels of the segmentation file should define what specific region of interest (ROI) each pixel belongs to. For example, if a total of 4 ROIs are being segmented, every pixel that belongs to ROI #1 should have a value of 1, every pixel that belongs to ROI #2 should have a value of 2, etc. and every pixel that is not labeled (i.e. belongs to the background or other regions) should have a value of 0.

WARNING: Make sure that the labels are consistent across samples! In other word, if the pixels of ROI #1 on image 1 have a value of 1, the pixels that belong to the same ROI on every other image in the training set should all have a value of 1.

### Notebook Setup
In the Settings section:
- Edit the different paths and filenames under the DIRECTORIES & FILENAMES section
- Choose the right input shape under the TRAINING PARAMETERS section
    - Optional: edit the save_checkpoint parameter to define how often to save the model during training
- Every other parameter can be left as their original value.

### Run the Notebook
Once the Settings have been edited, run each cell of the notebook and wait for training to be completed.


## References
[1] Dourthe B, Shaikh N, S AP, Fels S, Brown SHM, Wilson DR, Street J, Oxland TR. Automated Segmentation of spinal Muscles from Upright Open MRI Using a Multi-Scale Pyramid 2D Convolutional Neural Network. Spine (Phila Pa 1976). 2021 Dec 15. doi: 10.1097/BRS.0000000000004308. PMID: 34919072. https://pubmed.ncbi.nlm.nih.gov/34919072/

___
# Imports


```python
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# System characteristics
import psutil
import humanize
import GPUtil as GPU
import torch.backends.cudnn as cudnn

# Computation time monitoring
from time import time

# Data visualization
import matplotlib.pyplot as plt
from IPython.display import clear_output
from jupyterthemes import jtplot

# Data processing
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Pytoolbox import
from pytoolbox.utils import *
from pytoolbox.dataset import *
from pytoolbox.network import *
from pytoolbox.loss import *

print('Libraries successfully imported')
```

    Libraries successfully imported
    

___
# System Characteristics
The cell below allows you to check whether your GPU is enabled and displays the corresponding system characteristics such as memory usage.


```python
# Display local virtual memory (RAM)
print(f'\033[1mRAM Memory:\t\033[0m{humanize.naturalsize(psutil.virtual_memory().available)} (Available)\n')

# Check if GPU is enabled
print(f'\033[1mGPU enabled:\t\033[0m{torch.cuda.is_available()}\n')

# Setting device on GPU ('cuda') if available, if not, the device will be set to CPU ('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# If device set to GPU ('cuda'), display device information
if device.type == 'cuda':
    # Collect GPU information
    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    process = psutil.Process(os.getpid())
    print(f'\033[1mDevice Name:\t\033[0m{torch.cuda.get_device_name(0)}')
    print(f'\033[1mMemory Details:\t\033[0m{gpu.memoryTotal/1000:3.1f} GB '
          f'(Total)\t\t{gpu.memoryUsed/1000:3.1f} GB ({gpu.memoryUtil*100:.0f}% Used) '
          f'\t{gpu.memoryFree/1000:4.1f} GB ({100-gpu.memoryUtil*100:.0f}% Free)')
```

    RAM Memory:     23.3 GB (Available)
    
    GPU enabled:    True
    
    Device Name:    NVIDIA GeForce RTX 3080 Ti Laptop GPU
    Memory Details: 16.4 GB (Total)     0.1 GB (1% Used)    16.1 GB (99% Free)
    

___
# Settings


```python
###########################
# DIRECTORIES & FILENAMES #
###########################

# Define path towards data directory
main_dir = 'C:/Users/bdour/IDrive-Sync/Work/Academic/UBC/Spine Modeling Project/Data/Autosegmentation data/PublicRepo/data/open_MRI_ASD'

# Define paths towards training data
train_dicoms_path = main_dir + '/train/dicoms'
train_labels_path = main_dir + '/train/labels'

# Define path towards directory where the trained model and training history will be saved
model_export_path = 'trained_models'
# Check if the corresponding directory exists, if not, create it
if not os.path.exists(os.path.abspath(model_export_path)):
    os.mkdir(os.path.abspath(model_export_path))

# Define filename to save trained model state 
#   NOTE: the number of epochs will be added at the end of the model filename when saved
#   to reflect training state
model_filename = 'open_MR_autoseg_model'


#######################
# TRAINING PARAMETERS #
#######################

# Define input shape (i.e. number of pixels along x- or y-axis)
#   NOTE: Only one value is needed as input images are either squares,
#   or will be resized to squares within the data loader
input_shape = 256

# Define number of labels (i.e. number of regions to segment)
num_labels = 6

# Define number of epochs (i.e. iterations) used to complete training
epochs = 500

# Define data augmentation parameters
augment_params = {'aug_prob': 0.5,
                  'rotation': True,
                  'rot_range': 5,
                  'warping': True,
                  'warp_range': 0.8,
                  'cropping': True,
                  'crop_range': [0.5, 0.8]}

# Define computing parameters
# (i.e. how many samples are processed simultaneously and how many parallelized computers/GPUs are used)
batch_size = 5
num_workers = 0
pin_memory = False

# Define learning parameters
learning_rate = 0.0001
dropout_rate = 0.3

# Define checkpoints parameters
print_checkpoint = 10          # Defines how often a print statement will be displayed during training (in number of epochs)
save_checkpoint = 100          # Defines how often the model state will be saved (in number of epochs)


###############################
# DATA VISUALIZATION SETTINGS #
###############################

# Dark mode
dark_mode = True

# Define Jupyter theme based on dark mode
# list available themes
# onedork | grade3 | oceans16 | chesterish | monokai | solarizedl | solarizedd
if dark_mode:
    jtplot.style(theme='chesterish')
else:
    jtplot.style(theme='grade3')

print('Settings successfully defined')
```

    Settings successfully defined
    

___
# Dataset Pre-Visualization
The cell below allows you to load and display the first batch of the dataset after being passed through the data loader.

NOTE: This data loader includes data augmentation techniques that are implemented on-the-fly.


```python
###################
# DATASET LOADING #
###################

# Generate dataset
train_data = ProcessedDataset(train_dicoms_path, train_labels_path, input_shape=input_shape, augment_params=augment_params)

# Load dataset
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

# Grab the first sample
for img, lab in train_loader:
    break

# Print shape of the first batch
print('FIRST BATCH SHAPES:')
print(f'\tImages:\t{img.shape}')
print(f'\tLabels:\t{lab.shape}')

# Generate grids for low and high resolution images from first batch
#   Resize to 3D array and convert to numpy.ndarray 
img_arr = img.numpy()
lab_arr = lab.numpy()

#   Initialize grid arrays with a column vector of zeros
img_grid = np.zeros((img_arr.shape[3], 1))
lab_grid = np.zeros((lab_arr.shape[2], 1))
#   Isolate the indices of 5 evenly distanced labeled frames and 
for i in range(batch_size):
    # Stack each new image along the x-axis to generate a grid of 5 stacked images
    img_grid = np.hstack([img_grid, img_arr[i, 0, :, :]])
    lab_grid = np.hstack([lab_grid, lab_arr[i, :, :]])

# Display grids
plt.figure(figsize=(26, 6))
plt.imshow(img_grid, cmap='gray')
plt.imshow(lab_grid, alpha=0.6, cmap='cubehelix')
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.title('First Batch of Augmented Segmented Images', fontsize=20)
plt.show()
```

    FIRST BATCH SHAPES:
        Images: torch.Size([5, 1, 256, 256])
        Labels: torch.Size([5, 256, 256])
    


    
![png](img/output_8_1.png)
    


___
# Model Training


```python
######################
# DATASET DEFINITION #
######################

# Generate dataset
train_data = ProcessedDataset(train_dicoms_path, train_labels_path, augment_params=augment_params)

# Load dataset
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

# Grab the first sample
for img, lab in train_loader:
    break
    
# Define model parameters
input_dim = img.shape[1]
input_shape = img.shape[2]

########################
# MODEL INITIALIZATION #
########################

# Create initialization function
def init(model):
    if isinstance(model, nn.Conv3d) or isinstance(model, nn.ConvTranspose3d):
        nn.init.kaiming_normal(model.weight.data, 0.25)
        nn.init.constant(model.bias.data, 0)
        
# Create instance of the multi-scale pyramid model
model = MultiScalePyramid(num_labels=num_labels, input_shape=input_shape, training=True)

# Initialize model
model.apply(init)

############################
# OPTIMIZER INITIALIZATION #
############################

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define learning rate decay
lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)

##################
# MODEL TRAINING #
##################

# Assign device to 'cuda' if available, or to 'cpu' otherwise
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable cudnn benchmark, which allows cudnn to look for the optimal set of algorithms for the current device configuration
cudnn.benchmark = True

# Implements data parallelism at the module level
model = torch.nn.DataParallel(model)

# Assign model to available device ('cuda' or 'cpu')
model = model.to(device)

# Generate instance of the loss function
criterion = MultiClassDiceLoss(num_labels=num_labels, input_shape=input_shape)

# Initialize mean epoch loss tracker (-> one mean loss value per epoch)
mean_epoch_loss = []

# Initialize time tracker
start_time = time()

for epoch in range(1, epochs+1):
    
    # Update learning rate decay (will only update by gamma when epoch milestone is reached)
    lr_decay.step()
    
    # Initialize mean sample loss tracker (-> one mean loss value per sample)
    mean_sample_loss = []
    
    # Loop through training samples
    for img, lab in train_loader:
        
        # Assign image to available device ('cuda' or 'cpu')
        img = img.to(device)

        # Pass sample through model
        stage1_output, stage2_output = model(img)
        
        # Calculate sample loss
        loss = criterion(stage1_output, stage2_output, lab, device)
        
        # Append loss to mean sample loss tracker
        mean_sample_loss.append(loss.item())
        
        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Append loss to mean epoch loss tracker
    mean_epoch_loss.append(loss.item())    
    
    # Print update statement for tracking
    if epoch == 1 or epoch%print_checkpoint == 0:
        current_time = time() - start_time
        print(f'\033[1mCompleted epochs:\033[0m {epoch:4.0f}/{epochs} | '
              f'\033[1mDice loss:\033[0m {loss.item():4.3f} | '
              f'\033[1mMean dice score:\033[0m {(2-loss.item())/2:4.3f} | '
              f'\033[1mTime elapsed:\033[0m {current_time//3600:2.0f} hrs '
              f'{(current_time - current_time//3600*3600)//60:2.0f} mins '
              f'{current_time%60:2.0f} secs')

    # Save model every checkpoint
    if epoch%save_checkpoint == 0:
        #   Define current model state using dictionary
        model_state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        # Save model checkpoint
        torch.save(model_state, model_export_path + '/' + model_filename + '_' + str(model_state['epoch']) + '.pt')
        #   Print save statement
        print(f'\n\tCheckpoint -> Model saved at {epoch:4.0f}/{epochs} epochs\n')

# Define final model state using dictionary
model_state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

# Generate final loss history DataFrame
loss_history = pd.DataFrame(mean_epoch_loss, columns=['Dice loss'])

# Save final model and loss history
torch.save(model_state, model_export_path + '/' + model_filename + '_' + str(model_state['epoch']) + '.pt')
loss_history.to_csv(model_export_path + '/' + model_filename + '_history_' + str(epoch) + '.csv')

# Plot training history
loss_history.plot(figsize=(24, 6))
plt.legend(loc='upper right', fontsize=15)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('Dice loss', fontsize=20)
plt.title('Training history', fontsize=30)
plt.show()

# Print total computing time
total_time = time() - start_time
print(f'\033[1m\nTotal computing time:\033[0m {total_time//3600:2.0f} hrs '
      f'{(total_time - total_time//3600*3600)//60:2.0f} mins '
      f'{total_time%60:2.0f} secs')
```

    Completed epochs:    1/500 | Dice loss: 1.774 | Mean dice score: 0.113 | Time elapsed:  0 hrs  0 mins 10 secs
    Completed epochs:   10/500 | Dice loss: 1.262 | Mean dice score: 0.369 | Time elapsed:  0 hrs  0 mins 35 secs
    Completed epochs:   20/500 | Dice loss: 0.982 | Mean dice score: 0.509 | Time elapsed:  0 hrs  1 mins  4 secs
    Completed epochs:   30/500 | Dice loss: 1.021 | Mean dice score: 0.489 | Time elapsed:  0 hrs  1 mins 34 secs
    Completed epochs:   40/500 | Dice loss: 0.728 | Mean dice score: 0.636 | Time elapsed:  0 hrs  2 mins  3 secs
    Completed epochs:   50/500 | Dice loss: 0.753 | Mean dice score: 0.623 | Time elapsed:  0 hrs  2 mins 31 secs
    Completed epochs:   60/500 | Dice loss: 0.712 | Mean dice score: 0.644 | Time elapsed:  0 hrs  2 mins 60 secs
    Completed epochs:   70/500 | Dice loss: 0.575 | Mean dice score: 0.712 | Time elapsed:  0 hrs  3 mins 28 secs
    Completed epochs:   80/500 | Dice loss: 0.664 | Mean dice score: 0.668 | Time elapsed:  0 hrs  3 mins 56 secs
    Completed epochs:   90/500 | Dice loss: 0.662 | Mean dice score: 0.669 | Time elapsed:  0 hrs  4 mins 25 secs
    Completed epochs:  100/500 | Dice loss: 0.617 | Mean dice score: 0.691 | Time elapsed:  0 hrs  4 mins 53 secs
    
        Checkpoint -> Model saved at  100/500 epochs
    
    Completed epochs:  110/500 | Dice loss: 0.401 | Mean dice score: 0.800 | Time elapsed:  0 hrs  5 mins 20 secs
    Completed epochs:  120/500 | Dice loss: 0.606 | Mean dice score: 0.697 | Time elapsed:  0 hrs  5 mins 48 secs
    Completed epochs:  130/500 | Dice loss: 0.418 | Mean dice score: 0.791 | Time elapsed:  0 hrs  6 mins 15 secs
    Completed epochs:  140/500 | Dice loss: 0.344 | Mean dice score: 0.828 | Time elapsed:  0 hrs  6 mins 45 secs
    Completed epochs:  150/500 | Dice loss: 0.531 | Mean dice score: 0.734 | Time elapsed:  0 hrs  7 mins 13 secs
    Completed epochs:  160/500 | Dice loss: 0.509 | Mean dice score: 0.745 | Time elapsed:  0 hrs  7 mins 41 secs
    Completed epochs:  170/500 | Dice loss: 0.385 | Mean dice score: 0.808 | Time elapsed:  0 hrs  8 mins  9 secs
    Completed epochs:  180/500 | Dice loss: 0.356 | Mean dice score: 0.822 | Time elapsed:  0 hrs  8 mins 37 secs
    Completed epochs:  190/500 | Dice loss: 0.523 | Mean dice score: 0.739 | Time elapsed:  0 hrs  9 mins  4 secs
    Completed epochs:  200/500 | Dice loss: 0.453 | Mean dice score: 0.773 | Time elapsed:  0 hrs  9 mins 31 secs
    
        Checkpoint -> Model saved at  200/500 epochs
    
    Completed epochs:  210/500 | Dice loss: 0.603 | Mean dice score: 0.699 | Time elapsed:  0 hrs  9 mins 59 secs
    Completed epochs:  220/500 | Dice loss: 0.323 | Mean dice score: 0.839 | Time elapsed:  0 hrs 10 mins 27 secs
    Completed epochs:  230/500 | Dice loss: 0.378 | Mean dice score: 0.811 | Time elapsed:  0 hrs 10 mins 54 secs
    Completed epochs:  240/500 | Dice loss: 0.535 | Mean dice score: 0.733 | Time elapsed:  0 hrs 11 mins 22 secs
    Completed epochs:  250/500 | Dice loss: 0.258 | Mean dice score: 0.871 | Time elapsed:  0 hrs 11 mins 49 secs
    Completed epochs:  260/500 | Dice loss: 0.584 | Mean dice score: 0.708 | Time elapsed:  0 hrs 12 mins 16 secs
    Completed epochs:  270/500 | Dice loss: 0.636 | Mean dice score: 0.682 | Time elapsed:  0 hrs 12 mins 43 secs
    Completed epochs:  280/500 | Dice loss: 0.207 | Mean dice score: 0.896 | Time elapsed:  0 hrs 13 mins 11 secs
    Completed epochs:  290/500 | Dice loss: 0.291 | Mean dice score: 0.854 | Time elapsed:  0 hrs 13 mins 40 secs
    Completed epochs:  300/500 | Dice loss: 0.251 | Mean dice score: 0.875 | Time elapsed:  0 hrs 14 mins  7 secs
    
        Checkpoint -> Model saved at  300/500 epochs
    
    Completed epochs:  310/500 | Dice loss: 0.245 | Mean dice score: 0.877 | Time elapsed:  0 hrs 14 mins 34 secs
    Completed epochs:  320/500 | Dice loss: 0.272 | Mean dice score: 0.864 | Time elapsed:  0 hrs 15 mins  1 secs
    Completed epochs:  330/500 | Dice loss: 0.299 | Mean dice score: 0.851 | Time elapsed:  0 hrs 15 mins 26 secs
    Completed epochs:  340/500 | Dice loss: 0.329 | Mean dice score: 0.836 | Time elapsed:  0 hrs 15 mins 51 secs
    Completed epochs:  350/500 | Dice loss: 0.319 | Mean dice score: 0.841 | Time elapsed:  0 hrs 16 mins 17 secs
    Completed epochs:  360/500 | Dice loss: 0.485 | Mean dice score: 0.758 | Time elapsed:  0 hrs 16 mins 43 secs
    Completed epochs:  370/500 | Dice loss: 0.255 | Mean dice score: 0.873 | Time elapsed:  0 hrs 17 mins  8 secs
    Completed epochs:  380/500 | Dice loss: 0.236 | Mean dice score: 0.882 | Time elapsed:  0 hrs 17 mins 33 secs
    Completed epochs:  390/500 | Dice loss: 0.391 | Mean dice score: 0.805 | Time elapsed:  0 hrs 18 mins  0 secs
    Completed epochs:  400/500 | Dice loss: 0.194 | Mean dice score: 0.903 | Time elapsed:  0 hrs 18 mins 27 secs
    
        Checkpoint -> Model saved at  400/500 epochs
    
    Completed epochs:  410/500 | Dice loss: 0.174 | Mean dice score: 0.913 | Time elapsed:  0 hrs 18 mins 53 secs
    Completed epochs:  420/500 | Dice loss: 0.366 | Mean dice score: 0.817 | Time elapsed:  0 hrs 19 mins 20 secs
    Completed epochs:  430/500 | Dice loss: 0.351 | Mean dice score: 0.825 | Time elapsed:  0 hrs 19 mins 46 secs
    Completed epochs:  440/500 | Dice loss: 0.158 | Mean dice score: 0.921 | Time elapsed:  0 hrs 20 mins 15 secs
    Completed epochs:  450/500 | Dice loss: 0.197 | Mean dice score: 0.902 | Time elapsed:  0 hrs 20 mins 43 secs
    Completed epochs:  460/500 | Dice loss: 0.204 | Mean dice score: 0.898 | Time elapsed:  0 hrs 21 mins 11 secs
    Completed epochs:  470/500 | Dice loss: 0.692 | Mean dice score: 0.654 | Time elapsed:  0 hrs 21 mins 39 secs
    Completed epochs:  480/500 | Dice loss: 0.273 | Mean dice score: 0.864 | Time elapsed:  0 hrs 22 mins  6 secs
    Completed epochs:  490/500 | Dice loss: 0.540 | Mean dice score: 0.730 | Time elapsed:  0 hrs 22 mins 33 secs
    Completed epochs:  500/500 | Dice loss: 0.209 | Mean dice score: 0.896 | Time elapsed:  0 hrs 23 mins  1 secs
    
        Checkpoint -> Model saved at  500/500 epochs
    
    


    
![png](img/output_10_1.png)
    


        Total computing time:  0 hrs 23 mins  1 secs
    

<img src='img/icord_footer.jpg' /></a>
