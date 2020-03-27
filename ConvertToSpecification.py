import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import sys
from proto import network_pb2


PRE_TRAINED_DIR = "./pretrained-models/"

# Conv Layer Message
def makeConv2DMessage(conv_layer, layer_callback):
    kernel_size     = list(conv_layer.kernel_size)
    stride          = list(conv_layer.stride)
    padding         = list(conv_layer.padding)
    dilation        = list(conv_layer.dilation)

    weights = torch.flatten(conv_layer.weight.data).numpy().tolist()

    bias_present = False
    if conv_layer.bias is not None:
            bias_present = True
            bias = torch.flatten(conv_layer.bias.data).flatten().numpy().tolist()

    layer_callback.type                 = 1
    layer_callback.conv.out_channels    = conv_layer.out_channels
    layer_callback.conv.in_channels     = conv_layer.in_channels
    layer_callback.conv.height          = kernel_size[0]
    layer_callback.conv.width           = kernel_size[1]
    layer_callback.conv.stride          = stride[0]
    layer_callback.conv.padding         = padding[0]
    layer_callback.conv.dilation        = dilation[0]
    layer_callback.conv.weight[:]       = weights
    if bias_present:
        layer_callback.conv.bias_present = True
        layer_callback.conv.bias[:]      = bias


# Pooling Layer Message
def makePool2DMessage(pool_layer, layer_callback, avg = False, adaptive = False):
    if adaptive:
        out_size = list(pool_layer.output_size)

        layer_callback.type        = 3
        layer_callback.apool.type  = int(avg)
        layer_callback.apool.out_x = out_size[0]
        layer_callback.apool.out_y = out_size[1]
    else:
        layer_callback.type             = 2
        layer_callback.pool.type        = int(avg)
        layer_callback.pool.kernel_size = pool_layer.kernel_size
        layer_callback.pool.stride      = pool_layer.stride
        layer_callback.pool.padding     = pool_layer.padding
        layer_callback.pool.dilation    = pool_layer.dilation


# Linear Layer Message
def makeFCMessage(fc_layer, layer_callback):
    in_features     = fc_layer.in_features
    out_features    = fc_layer.out_features
    weights         = torch.flatten(fc_layer.weight.data).numpy().tolist()
    
    bias_present = False
    if fc_layer.bias is not None:
        bias_present = True
        bias = torch.flatten(fc_layer.bias.data).numpy().tolist()

    layer_callback.type = 0
    layer_callback.linear.in_nodes  = in_features
    layer_callback.linear.out_nodes = out_features
    layer_callback.linear.weight[:] = weights

    if bias_present:
        layer_callback.linear.bias_present = True
        layer_callback.linear.bias[:]      = bias


# ReLU Activation Message
def makeReLUMessage(layer_callback):
    layer_callback.type = 5
    layer_callback.act.type = 0


# DropOut Layer Message
def makeDropoutMessage(dropout_layer, layer_callback):
    layer_callback.type     = 4
    layer_callback.drop.p   = dropout_layer.p 


def createProtoSpecification(model, filename):
    net = network_pb2.Network()
    num_layers = 0

    # Iterate through the layers #
    for layer in model.children():
            for child in layer.modules():
                if isinstance(child ,nn.Conv2d):
                    #Make the conv message
                    convLayer = net.layers.add() 
                    makeConv2DMessage(child, convLayer)
                    num_layers+=1
                
                elif isinstance(child,nn.MaxPool2d):
                    #Make the pool message
                    poolLayer = net.layers.add()
                    makePool2DMessage(child,poolLayer)
                    num_layers+=1
                
                elif isinstance(child,nn.AdaptiveAvgPool2d):
                    #Make the adaptive pool message
                    apoolLayer = net.layers.add()
                    makePool2DMessage(child, apoolLayer, avg=True, adaptive=True)
                    num_layers+=1
                
                elif isinstance(child,nn.ReLU):
                    #Make the activation message
                    reluact = net.layers.add()
                    makeReLUMessage(reluact)
                    num_layers+=1

                elif isinstance(child,nn.Linear):
                    #Make the linear layer message
                    linearLayer = net.layers.add()
                    makeFCMessage(child, linearLayer)
                    num_layers+=1

                elif isinstance(child,nn.Dropout):
                    #Make the DropOut layer message
                    dropLayer = net.layers.add()
                    makeDropoutMessage(child, dropLayer)
                    num_layers+=1
    
    net.num_layers = num_layers

    #Store in Pre-trained Models
    filename = PRE_TRAINED_DIR+filename
    f = open(filename, "wb")
    f.write(net.SerializeToString())
    f.close()

def createVGGSpecification(filename):
    ## VGG Model ##
    vgg19 = models.vgg19(pretrained=True)
    createProtoSpecification(vgg19, filename)

def createAlexNetSpecification(filename):
    ## VGG Model ##
    alex = models.alexnet(pretrained=True)
    createProtoSpecification(alex, filename)

if __name__ == "__main__":
    createVGGSpecification("vgg19.pb")
    createAlexNetSpecification("alexnet.pb")

    