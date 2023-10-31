import torch
from torch.functional import Tensor
import torch.nn as nn

""" This script defines the network.
"""

class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsample block layer.
        
        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        # filter_num = num output channels

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.
        
        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        # define conv1

        # conv(1x1) - map 3 RGB to 16 channel
        self.start_layer = nn.Conv2d(in_channels=3,
                                     out_channels= first_num_filters, 
                                     kernel_size=(1,1), 
                                     stride=1)
        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
                eps=1e-5, 
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        for i in range(3):
            in_channel = out_channel if i != 0 else self.first_num_filters
            
            if self.resnet_version == 1:
                out_channel = self.first_num_filters * (2**i)
            else:
                out_channel = in_channel*4

            strides = 1 if i == 0 else 2

            self.stack_layers.append(stack_layer(out_channel, block_fn, strides, self.resnet_size, in_channel))
        self.output_layer = output_layer(out_channel, self.resnet_version, self.num_classes)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)

        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)

        for i in range(3):
            outputs = self.stack_layers[i](outputs)

        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        ### YOUR CODE HERE
    def forward(self, inputs: Tensor) -> Tensor:
        """
        # input size 4d
        (N, C, H, W)
        # output size 4d
        """
        ### YOUR CODE HERE
        output = self.batch_norm(inputs)
        output = self.relu(output)
        ### YOUR CODE HERE
        return output
    
class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when down-sampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately down-sample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE

        ## only use stride for first layer

        ### first_num_filters = in_channel
        ### filters = out_channel

        in_channels = first_num_filters
        out_channels = filters

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(3,3),
                      stride=strides, # decrease dimension
                      padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=out_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
        ### YOUR CODE HERE
        
        ### YOUR CODE HERE

        # if first block, using projection shortcut
        self.projection_shortcut = None
        if projection_shortcut:
            self.projection_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels= out_channels, 
                      kernel_size=(1,1), 
                      stride=strides),
            nn.BatchNorm2d(num_features=out_channels)
        )

        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE

        if self.projection_shortcut is not None:
            shortcut = self.projection_shortcut(inputs)
        else:
            shortcut = inputs

        output = self.block(inputs)
        output += shortcut
        output = self.relu(output)
        return output            
        ### YOUR CODE HERE

class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4*filters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when down-sampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately down-sample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()

        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to 
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.

        in_channels = first_num_filters
        intermediate_channels = filters/4
        out_channels = filters


        self.block = nn.Sequential(
        nn.BatchNorm2d(num_features=in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=in_channels, 
                    out_channels=intermediate_channels, 
                    kernel_size=(1, 1), 
                    stride=1),
    
        nn.BatchNorm2d(num_features=intermediate_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=intermediate_channels, 
                    out_channels=intermediate_channels, 
                    kernel_size=(3, 3), 
                    stride=strides, 
                    padding=1),
    
        nn.BatchNorm2d(num_features=intermediate_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=intermediate_channels, 
                    out_channels=out_channels, 
                    kernel_size=(1, 1), 
                    stride=1)
        )
        
        # if first block, using projection shortcut
        self.projection_shortcut = None
        if projection_shortcut:
            self.projection_shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=(1, 1), 
                        stride=strides),
                nn.BatchNorm2d(num_features=out_channels)
            )
        ### YOUR CODE HERE

    
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.

        if self.projection_shortcut is not None:
            shortcut = self.projection_shortcut(inputs)
        else:
            shortcut = inputs

        output = self.block(inputs)
        output += shortcut

        return output 
        ### YOUR CODE HERE

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately down-sample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()

        # I would define in channels and out channels at the ResNet class 
        # filters_out = filters * 4 if block_fn is bottleneck_block else filters
        ### END CODE HERE
        # projection_shortcut = bool variable, would create function inside the block class if True
        
        in_channels = first_num_filters
        out_channels = filters
        self.stack = nn.Sequential()

        # Only the first block per stack_layer uses projection_shortcut and strides
        self.stack.add_module("block1", block_fn(filters=out_channels, 
                                                  projection_shortcut=True, 
                                                  strides=strides, 
                                                  first_num_filters=in_channels))
        
        
        for i in range(2, resnet_size+1):
            self.stack.add_module(f"block{i}", block_fn(filters=out_channels, 
                                                    projection_shortcut=False, 
                                                    strides=1, 
                                                    first_num_filters=out_channels))
            
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        output = self.stack(inputs)
        return output
        ### END CODE HERE

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)
        
        ### END CODE HERE
        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=filters, 
                            out_features=num_classes)
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        if self.resnet_version == 2:
            inputs = self.bn_relu(inputs)
        
        output = self.global_avg_pool(inputs)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        
        return output
        ### END CODE HERE