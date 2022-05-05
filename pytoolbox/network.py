####################
# LIBRARIES IMPORT #
####################

import torch
import torch.nn as nn
import torch.nn.functional as F

####################
# CLASS DEFINITION #
####################

#----------------------------------------------------------------------------------------------------#
class MultiLabelVNet(nn.Module):
    
    def __init__(self, stage, num_labels, input_shape=256, num_channels=1, dropout_rate=0.3, training=False):
        '''
        DOURTHE TECHNOLOGIES - CONFIDENTIAL
	    Unpublished Copyright © 2022 Dourthe Technologies (dourthetechnologies.com)

	    Created on: Tue Apr 12 2022 
	    Author: Benjamin Dourthe (benjamin@dourthe.tech)

	    Description:
	    ------------

            Multiclass adaptation of the V-Net Fully Convolutional Network (FCN) developed by
            Milletari et al. (2016).
	            Milletari F, Navab N, Ahmadi SA. V-Net: Fully Convolutional Neural Networks
	            for Volumetric Medical Image Segmentation.
	            arXiv:1606.04797v1, 2016 (https://arxiv.org/abs/1606.04797)

        Parameters:
        -----------

			stage: specifies current stage.
                (type: string - 'stage 1' for 1st pass through 3D FCN, 'stage 2' for 2nd pass)
            num_labels: specifies number of labels.
            	(type: integer)
            input_shape: specifies shape of input image (e.g. input_shape=128 for input images of size 128x128).
                (type: integer) (default=256)
            num_channels: specifies number of input channels.
               (type: integer) (default=1)
            dropout_out: define dropout rate.
            	(type: float) (default=0.3)
            training: specifies whether the network is currently being used for training or testing.
                (type: boolean - True for training, False for testing) (default=False)          
        '''
        
        super().__init__()
        
        self.stage = stage
        self.num_labels = num_labels
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.training = training
        
        # ENCODER        
        self.encoder1 = nn.Sequential(nn.Conv2d(self.num_channels,
        										int(self.input_shape/16),
        										3, 1, padding=1),
                                      nn.PReLU(int(self.input_shape/16)))   
        self.encoder2 = nn.Sequential(nn.Conv2d(int(self.input_shape/8),
        										int(self.input_shape/8),
        										3, 1, padding=1),
                                      nn.PReLU(int(self.input_shape/8)),
                                      nn.Conv2d(int(self.input_shape/8),
                                      			int(self.input_shape/8),
                                      			3, 1, padding=1),
                                      nn.PReLU(int(self.input_shape/8)))        
        self.encoder3 = nn.Sequential(nn.Conv2d(int(self.input_shape/4),
        										int(self.input_shape/4),
        										3, 1, padding=1),
                                      nn.PReLU(int(self.input_shape/4)),
                                      nn.Conv2d(int(self.input_shape/4),
                                      			int(self.input_shape/4),
                                      			3, 1, padding=2, dilation=2),
                                      nn.PReLU(int(self.input_shape/4)),
                                      nn.Conv2d(int(self.input_shape/4),
                                      			int(self.input_shape/4),
                                      			3, 1, padding=4, dilation=4),
                                      nn.PReLU(int(self.input_shape/4)))        
        self.encoder4 = nn.Sequential(nn.Conv2d(int(self.input_shape/2),
        										int(self.input_shape/2),
        										3, 1, padding=3, dilation=3),
                                      nn.PReLU(int(self.input_shape/2)),
                                      nn.Conv2d(int(self.input_shape/2),
                                      			int(self.input_shape/2),
                                      			3, 1, padding=4, dilation=4),
                                      nn.PReLU(int(self.input_shape/2)),
                                      nn.Conv2d(int(self.input_shape/2),
                                      			int(self.input_shape/2),
                                      			3, 1, padding=5, dilation=5),
                                      nn.PReLU(int(self.input_shape/2)))
        
        # DECODER        
        self.decoder1 = nn.Sequential(nn.Conv2d(int(self.input_shape/2),
        										self.input_shape,
        										3, 1, padding=1),
                                      nn.PReLU(self.input_shape),
                                      nn.Conv2d(self.input_shape,
                                      			self.input_shape,
                                      			3, 1, padding=1),
                                      nn.PReLU(self.input_shape),
                                      nn.Conv2d(self.input_shape,
                                      			self.input_shape,
                                      			3, 1, padding=1),
                                      nn.PReLU(self.input_shape))        
        self.decoder2 = nn.Sequential(nn.Conv2d(int(self.input_shape/2)+int(self.input_shape/4),
        										int(self.input_shape/2),
        										3, 1, padding=1),
                                      nn.PReLU(int(self.input_shape/2)),
                                      nn.Conv2d(int(self.input_shape/2),
                                      			int(self.input_shape/2),
                                      			3, 1, padding=1),
                                      nn.PReLU(int(self.input_shape/2)),
                                      nn.Conv2d(int(self.input_shape/2),
                                      			int(self.input_shape/2),
                                      			3, 1, padding=1),
                                      nn.PReLU(int(self.input_shape/2)))        
        self.decoder3 = nn.Sequential(nn.Conv2d(int(self.input_shape/4)+int(self.input_shape/8),
        										int(self.input_shape/4),
        										3, 1, padding=1),
                                      nn.PReLU(int(self.input_shape/4)),
                                      nn.Conv2d(int(self.input_shape/4),
                                      			int(self.input_shape/4),
                                      			3, 1, padding=1),
                                      nn.PReLU(int(self.input_shape/4)))        
        self.decoder4 = nn.Sequential(nn.Conv2d(int(self.input_shape/8)+int(self.input_shape/16),
        										int(self.input_shape/8),
        										3, 1, padding=1),
                                      nn.PReLU(int(self.input_shape/8)))
        
        # DOWN CONVOLUTIONS     
        self.down_conv1 = nn.Sequential(nn.Conv2d(int(self.input_shape/16),
        										  int(self.input_shape/8),
        										  2, 2),
                                        nn.PReLU(int(self.input_shape/8)))        
        self.down_conv2 = nn.Sequential(nn.Conv2d(int(self.input_shape/8),
        										  int(self.input_shape/4),
        										  2, 2),
                                        nn.PReLU(int(self.input_shape/4)))        
        self.down_conv3 = nn.Sequential(nn.Conv2d(int(self.input_shape/4),
        										  int(self.input_shape/2),
        										  2, 2),
                                        nn.PReLU(int(self.input_shape/2)))       
        self.down_conv4 = nn.Sequential(nn.Conv2d(int(self.input_shape/2),
        										  self.input_shape,
        										  3, 1, padding=1),
                                        nn.PReLU(self.input_shape))
        
        # UP CONVOLUTIONS     
        self.up_conv1 = nn.Sequential(nn.ConvTranspose2d(self.input_shape,
        												 int(self.input_shape/2),
        												 2, 2),
                                      nn.PReLU(int(self.input_shape/2)))        
        self.up_conv2 = nn.Sequential(nn.ConvTranspose2d(int(self.input_shape/2),
        												 int(self.input_shape/4),
        												 2, 2),
                                      nn.PReLU(int(self.input_shape/4)))        
        self.up_conv3 = nn.Sequential(nn.ConvTranspose2d(int(self.input_shape/4),
        												 int(self.input_shape/8),
        												 2, 2),
                                      nn.PReLU(int(self.input_shape/8)))
        
        # OUTPUT        
        self.map = nn.Sequential(nn.Conv2d(int(self.input_shape/8),
        								   self.num_labels+1, 1),
        					 	 nn.Softmax(dim=1))
        
    def forward(self, inputs):
        
        # ENCODER + DOWN CONVOLUTIONS
        if self.stage == 'stage1':
            enc_layer1 = self.encoder1(inputs) + inputs
        elif self.stage == 'stage2':
            enc_layer1 = self.encoder1(inputs)            
        down1 = self.down_conv1(enc_layer1)
        
        enc_layer2 = self.encoder2(down1) + down1
        enc_layer2 = F.dropout(enc_layer2, self.dropout_rate, self.training)        
        down2 = self.down_conv2(enc_layer2)
        
        enc_layer3 = self.encoder3(down2) + down2
        enc_layer3 = F.dropout(enc_layer3, self.dropout_rate, self.training)        
        down3 = self.down_conv3(enc_layer3)
        
        enc_layer4 = self.encoder4(down3) + down3
        enc_layer4 = F.dropout(enc_layer4, self.dropout_rate, self.training)        
        down4 = self.down_conv4(enc_layer4)
        
        # DECODER + UP CONVOLUTIONS
        dec_layer1 = self.decoder1(enc_layer4) + down4
        dec_layer1 = F.dropout(dec_layer1, self.dropout_rate, self.training)        
        up1 = self.up_conv1(dec_layer1)
        
        dec_layer2 = self.decoder2(torch.cat([up1, enc_layer3], dim=1)) + up1
        dec_layer2 = F.dropout(dec_layer2, self.dropout_rate, self.training)        
        up2 = self.up_conv2(dec_layer2)
        
        dec_layer3 = self.decoder3(torch.cat([up2, enc_layer2], dim=1)) + up2
        dec_layer3 = F.dropout(dec_layer3, self.dropout_rate, self.training)        
        up3 = self.up_conv3(dec_layer3)
        
        dec_layer4 = self.decoder4(torch.cat([up3, enc_layer1], dim=1)) + up3
        
        # OUTPUT
        output = self.map(dec_layer4)
        
        return output

#----------------------------------------------------------------------------------------------------#
class MultiScalePyramid(nn.Module):
    
    def __init__(self, num_labels, input_shape=256, dropout_rate=0.3, training=False):
        '''
        DOURTHE TECHNOLOGIES - CONFIDENTIAL
	    Unpublished Copyright © 2022 Dourthe Technologies (dourthetechnologies.com)

	    Created on: Tue Apr 12 2022 
	    Author: Benjamin Dourthe (benjamin@dourthe.tech)

	    Description:
	    ------------

            Multi-scale Pyramid Fully Convolutional Network (FCN) implementation based on the work of
            Roth et al. (2018).
            	Roth HR, Shen C, Oda H, Sugino T, Oda M, Hayashi Y, Misawa K, Mori K.
            	A multi-scale pyramid of 3D fully convolutional networks for abdominal
            	multi-organ segmentation.
            	arXiv:1806.02237v1, 2018 (https://arxiv.org/abs/1806.02237)

        Parameters:
        -----------

			num_labels: number of labels.
				(type: integer)			
			input_shape: shape of the input image along a single dimension (assumed squared image).
				(type: integer) (default=256)
			dropout_out: define dropout rate
            	(type: float) (default=0.3)
			training: specifies whether the network is currently being used for training or testing
                (type: boolean - True for training, False for testing) (default=False)
            
        '''

        super().__init__()
        
        self.num_labels = num_labels
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.training = training
        
        self.stage1 = MultiLabelVNet(stage='stage1',
        							 num_labels=self.num_labels,
        							 input_shape=self.input_shape,
                                     num_channels=1,
                                     dropout_rate=self.dropout_rate,
        							 training=self.training)

        self.stage2 = MultiLabelVNet(stage='stage2',
        							 num_labels=self.num_labels,
        							 input_shape=self.input_shape,
                                     num_channels=self.num_labels+2,
                                     dropout_rate=self.dropout_rate,
                                     training=self.training)
        
    def forward(self, inputs):
        
        # STAGE 1
        #   Downsample original images by ds1 = 2S (i.e. decrease resolution by a factor of 2)
        stage1_input = F.upsample(inputs, (int(self.input_shape/2), int(self.input_shape/2)), mode='bilinear')        
        #   Generate stage 1 output by passing downsampled input through the 3D FCN
        stage1_output = self.stage1(stage1_input)        
        #   Upsample output back to original size
        stage1_output = F.upsample(stage1_output, (self.input_shape, self.input_shape), mode='bilinear')
        
        # STAGE 2
        #   Concatenate stage 1 output with original image
        stage2_input = torch.cat((stage1_output, inputs), dim=1)
        #   Generate stage 2 output by passing stage 2 input through the 3D FCN
        stage2_output = self.stage2(stage2_input)
        
        # Return results based on training status
        if self.training == True:
            return stage1_output, stage2_output
        else:
            return stage2_output