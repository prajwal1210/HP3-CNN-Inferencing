#################################################################################
#  ConvertToSpecification.py:                                                   #
#  Loads pretrained VGG-19 and AlexNet models in Pytorch and saves them         #
#  according to the custom protobuf format specified under proto/network.proto  #
#                                                                               #
#  Author: Prajwal Singhania                                                    #
#################################################################################

import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import sys
from proto import network_pb2

PRE_TRAINED_DIR = "./pretrained-models/"


# Conv Layer Message
def makeConv2DMessage(conv_layer, layer_callback):
    """
    Creates a Conv2D message in the specified protobuf format    
        Input : Pytorch nn.Conv2D object, Layer Object (defined in the proto specification) in which to store details of the Conv2D layer
        Result : Adds the appropriate parameters to the layer_callback object
    """
    kernel_size = list(conv_layer.kernel_size)
    stride = list(conv_layer.stride)
    padding = list(conv_layer.padding)
    dilation = list(conv_layer.dilation)

    weights = torch.flatten(conv_layer.weight.data).numpy().tolist()

    bias_present = False
    if conv_layer.bias is not None:
        bias_present = True
        bias = torch.flatten(conv_layer.bias.data).flatten().numpy().tolist()

    layer_callback.type = 1
    layer_callback.conv.out_channels = conv_layer.out_channels
    layer_callback.conv.in_channels = conv_layer.in_channels
    layer_callback.conv.height = kernel_size[0]
    layer_callback.conv.width = kernel_size[1]
    layer_callback.conv.stride = stride[0]
    layer_callback.conv.padding = padding[0]
    layer_callback.conv.dilation = dilation[0]
    layer_callback.conv.weight[:] = weights
    if bias_present:
        layer_callback.conv.bias_present = True
        layer_callback.conv.bias[:] = bias
    else:
        layer_callback.conv.bias_present = False


# Pooling Layer Message
def makePool2DMessage(pool_layer, layer_callback, avg=False, adaptive=False):
    """
    Creates a Pool message in the specified protobuf format - Supports both Normal and Adaptive Pooling (Max and Average)
        Input : Pytorch nn.Pool2D object or nn.AdaptivePool2D,  Layer Object (defined in the proto specification) in which to store details of the Pool2D layer
        Result : Adds the appropriate parameters to the layer_callback object
    """
    if adaptive:
        out_size = list(pool_layer.output_size)

        layer_callback.type = 3
        layer_callback.apool.type = int(avg)
        layer_callback.apool.out_x = out_size[0]
        layer_callback.apool.out_y = out_size[1]
    else:
        layer_callback.type = 2
        layer_callback.pool.type = int(avg)
        layer_callback.pool.kernel_size = pool_layer.kernel_size
        layer_callback.pool.stride = pool_layer.stride
        layer_callback.pool.padding = pool_layer.padding
        if (avg):
            layer_callback.pool.dilation = 1
        else:
            layer_callback.pool.dilation = pool_layer.dilation


# Linear Layer Message
def makeFCMessage(fc_layer, layer_callback):
    """
    Creates a Linear Layer (FC) message in the specified protobuf format
        Input : Pytorch nn.Linear object,  Layer Object (defined in the proto specification) in which to store details of the Linear layer
        Result : Adds the appropriate parameters to the layer_callback object
    """
    in_features = fc_layer.in_features
    out_features = fc_layer.out_features
    weights = torch.flatten(fc_layer.weight.data).numpy().tolist()

    bias_present = False
    if fc_layer.bias is not None:
        bias_present = True
        bias = torch.flatten(fc_layer.bias.data).numpy().tolist()

    layer_callback.type = 0
    layer_callback.linear.in_nodes = in_features
    layer_callback.linear.out_nodes = out_features
    # Weights are stored in out_features X in_features format
    layer_callback.linear.weight[:] = weights

    if bias_present:
        layer_callback.linear.bias_present = True
        layer_callback.linear.bias[:] = bias
    else:
        layer_callback.linear.bias_present = False


# ReLU Activation Message
def makeReLUMessage(layer_callback):
    """
    Creates a ReLU Activation message in the specified protobuf format 
        Input : Pytorch nn.ReLU Object,  Layer Object (defined in the proto specification) in which to store details of the Activation layer
        Result : Adds the appropriate parameters to the layer_callback object
    """
    layer_callback.type = 5
    layer_callback.act.type = 0

# Sigmoif Activation Message
def makeSigmoidMessage(layer_callback):
    """
    Creates a Sigmoid Activation message in the specified protobuf format 
        Input : Pytorch nn.ReLU Object,  Layer Object (defined in the proto specification) in which to store details of the Activation layer
        Result : Adds the appropriate parameters to the layer_callback object
    """
    layer_callback.type = 5
    layer_callback.act.type = 1

# DropOut Layer Message
def makeDropoutMessage(dropout_layer, layer_callback):
    """
    Creates a Dropout message in the specified protobuf format 
        Input : Pytorch nn.Dropout Object,  Layer Object (defined in the proto specification) in which to store details of the Dropout layer
        Result : Adds the appropriate parameters to the layer_callback object
    """
    layer_callback.type = 4
    layer_callback.drop.p = dropout_layer.p


def createProtoSpecification(model, filename):
    """
    Creates and saves the Network information in a .pb file based the specified protobuf format 
        Input : Pytorch model (pre-trained model), Name of the file in which to store the model message
        Result : Saves the model with the given filename under the directory - PRE_TRAINED_DIR
    """
    net = network_pb2.Network()
    num_layers = 0

    # Iterate through the layers
    for layer in model.children():
        for child in layer.modules():
            if isinstance(child, nn.Conv2d):
                # Make the conv message
                convLayer = net.layers.add()
                makeConv2DMessage(child, convLayer)
                num_layers += 1

            elif isinstance(child, nn.MaxPool2d):
                # Make the pool message
                poolLayer = net.layers.add()
                makePool2DMessage(child, poolLayer)
                num_layers += 1
            
            elif isinstance(child, nn.AvgPool2d):
                # Make the pool message
                poolLayer = net.layers.add()
                makePool2DMessage(child, poolLayer, avg=True)
                num_layers += 1

            elif isinstance(child, nn.AdaptiveAvgPool2d):
                # Make the adaptive pool message
                apoolLayer = net.layers.add()
                makePool2DMessage(child, apoolLayer, avg=True, adaptive=True)
                num_layers += 1

            elif isinstance(child, nn.ReLU):
                # Make the activation message
                reluact = net.layers.add()
                makeReLUMessage(reluact)
                num_layers += 1

            elif isinstance(child, nn.Sigmoid):
                # Make the activation message
                sigact = net.layers.add()
                makeSigmoidMessage(sigact)
                num_layers += 1

            elif isinstance(child, nn.Linear):
                # Make the linear layer message
                linearLayer = net.layers.add()
                makeFCMessage(child, linearLayer)
                num_layers += 1

            elif isinstance(child, nn.Dropout):
                # Make the DropOut layer message
                dropLayer = net.layers.add()
                makeDropoutMessage(child, dropLayer)
                num_layers += 1

    net.num_layers = num_layers

    # Store in Pre-trained Models
    filename = PRE_TRAINED_DIR + filename
    f = open(filename, "wb")
    f.write(net.SerializeToString())
    f.close()


def createVGGSpecification(filename):
    """
    Saves the pretrained VGG-19 Model in the specified protobuf format 
        Input : Name of the file in which to store the model message
        Result : Saves the model with the given filename under the directory - PRE_TRAINED_DIR
    """
    vgg19 = models.vgg19(pretrained=True)
    createProtoSpecification(vgg19, filename)


def createAlexNetSpecification(filename):
    """
    Saves the pretrained AlexNet Model in the specified protobuf format 
        Input : Name of the file in which to store the model message
        Result : Saves the model with the given filename under the directory - PRE_TRAINED_DIR
    """
    alex = models.alexnet(pretrained=True)
    createProtoSpecification(alex, filename)

def createTestModeSpecification(filename):
    """
    Creates and saves a Toy Model for Testing in the specified protobuf format 
        Input : Name of the file in which to store the model message
        Result : Saves the model with the given filename under the directory - PRE_TRAINED_DIR
    """
    test_model = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=(3,3), padding=1, bias=False),
    nn.AvgPool2d(4),
    nn.Sigmoid(),
    nn.Flatten(),
    nn.Linear(37674,100)
    )

    #Create the custom Kernel
    kernel_template = [[1,  1 , 1],[1, -8 , 1],[1,  1, 1]]

    h_kernel = []
    for k in range(3):
        temp_k = []
        for c in range(3):
            temp_k.append(kernel_template)
        h_kernel.append(temp_k)

    h_kernel = np.asarray(h_kernel)
    
    #Create weights
    out_nodes = 100
    in_nodes = 37674
    MAX = 1024
    k = 0
    w = []
    for i in range(out_nodes):
        row = []
        for j in range(in_nodes):
            row.append(k%MAX)
            k+=1
        w.append(row)
    w = np.asarray(w)
    b = [x%MAX for x in range(out_nodes)]
    b = np.asarray(b)

    #Set weights to layers
    with torch.no_grad():
        test_model[0].weight = nn.Parameter(torch.from_numpy(h_kernel).float())
        test_model[4].weight = nn.Parameter(torch.from_numpy(w).float())
        test_model[4].bias = nn.Parameter(torch.from_numpy(b).float())

    createProtoSpecification(test_model, filename)


if __name__ == "__main__":
    createVGGSpecification("vgg19.pb")
    createAlexNetSpecification("alexnet.pb")
    # createTestModeSpecification("test.pb")
