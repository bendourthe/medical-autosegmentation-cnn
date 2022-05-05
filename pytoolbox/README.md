# Python Toolbox
___

# Content

## Dataset (dataset.py)

Class/Function Name | Description |
--- | --- |
ProcessedDataset | Class enabling the definition of a dataset with on-the-fly data augmentation for the training of a Multiscale Pyramid 2D Convolutional Neural Network aiming at performing the automated segmentation of Magnetic Resonance Images.|

## Loss function (loss.py)

Class/Function Name | Description |
--- | --- |
MultiClassDiceLoss | Multiclass adaptation of the average Dice loss originally presented by Milletari et al. (2016).|

## Neural Network (network.py)

Class/Function Name | Description |
--- | --- |
MultiLabelVNet | Multiclass adaptation of the V-Net Fully Convolutional Network (FCN) developed by Milletari et al. (2016).|
MultiScalePyramid | Multi-scale Pyramid Fully Convolutional Network (FCN) implementation based on the work of Roth et al. (2018).|

## Utility functions (utils.py)

Class/Function Name | Description |
--- | --- |
display_dicom | Displays selected DICOM along with the corresponding DICOM information.|
preprocess_dicom | Loads and converts DICOM to normalized numpy array with selected data type.
