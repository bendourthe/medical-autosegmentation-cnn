####################
# LIBRARIES IMPORT #
####################

import numpy as np
import matplotlib.pyplot as plt
import pydicom
from dicompylercore import dicomparser

########################
# FUNCTIONS DEFINITION #
########################

#----------------------------------------------------------------------------------------------------#
def display_dicom(path):
    '''
    DOURTHE TECHNOLOGIES - CONFIDENTIAL
    Unpublished Copyright © 2022 Dourthe Technologies (dourthetechnologies.com)

    Created on: Mon Apr 11 2022 
    Author: Benjamin Dourthe (benjamin@dourthe.tech)

    Description:
    ------------

        Displays selected DICOM along with the corresponding DICOM information.

    Parameters:
    -----------

        path: complete path towards selected DICOM file.
            (type: string)
    '''

    # Read selected DICOM
    dicom = pydicom.dcmread(path)
        
    # Print DICOM information
    print(f'\033[1mModality:\033[0m\t {dicom.Modality}')
    print(f'\033[1mPatient ID:\033[0m\t {dicom.PatientID}')
    print(f'\033[1mStudy date:\033[0m\t {dicom.StudyDate[:4]}-{dicom.StudyDate[4:6]}-{dicom.StudyDate[6:]}')
    print(f'\033[1mFile size:\033[0m\t {str(len(dicom.PixelData))[:3]},{str(len(dicom.PixelData))[3:]} bytes')
    print(f'\033[1mImage size:\033[0m\t {int(dicom.Rows)} x {int(dicom.Columns)}')        
    print(f'\033[1mPixel spacing:\033[0m\t {dicom.PixelSpacing[0]} mm')

    # Display selected DICOM
    plt.figure(figsize=(10, 10))
    plt.imshow(dicom.pixel_array, cmap='gray')
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.show()

#----------------------------------------------------------------------------------------------------#
def preprocess_dicom(path, dtype=np.int32):
    '''
    DOURTHE TECHNOLOGIES - CONFIDENTIAL
    Unpublished Copyright © 2022 Dourthe Technologies (dourthetechnologies.com)

    Created on: Mon Apr 11 2022 
    Author: Benjamin Dourthe (benjamin@dourthe.tech)

    Description:
    ------------

        Loads and converts DICOM to normalized numpy array with selected data type.

    Parameters:
    -----------

        path: complete path towards selected DICOM file.
            (type: string)
        dtype: type of data to use during numpy array conversion.
            (type: numpy dtype) (default=np.int32)

    Returns:
    --------

        dicom_arr: DICOM file converted into a normalized numpy array.
            (type: array)
    '''
    # Read DICOM using secondary library (which allows image normalization)
    dicom = dicomparser.DicomParser(path)
    
    # Convert DICOM file to numpy array and 
    dicom_arr = np.array(dicom.GetImage())
    
    # Convert array to selected format
    dicom_arr = dicom_arr.astype(dtype) 

    # Normalize pixel values between 0 and 1
    dicom_arr = dicom_arr + np.abs(dicom_arr.min())
    dicom_arr = dicom_arr/(dicom_arr.max())
    
    return dicom_arr   