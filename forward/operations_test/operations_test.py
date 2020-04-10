import torch
import torch.nn as nn
import numpy as np
import sys
import cv2

#Load the Image
def loadImage(imagePath):
    img = cv2.imread(imagePath,cv2.IMREAD_COLOR).astype(np.float32)
    res = img
    res = cv2.normalize(res, res, 0.0, 1.0, cv2.NORM_MINMAX)
    return res

#Show the Image
def showImage(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Create the kernel
def createKernel():
    kernel_template = [[1,  1 , 1],[1, -8 , 1],[1,  1, 1]]

    h_kernel = []
    for k in range(3):
        temp_k = [] 
        for c in range(3):
            temp_k.append(kernel_template)
        h_kernel.append(temp_k)

    h_kernel = np.asarray(h_kernel)
    return h_kernel

#Create Weight and Bias for the Linear Layer Testing
def createWeightAndBias(out_nodes, in_nodes):
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
    return w,b

#Post process the image after running the model
def post_process(image):
    _, res = cv2.threshold(image, 0, 0, cv2.THRESH_TOZERO)
    res = res.astype(np.float32)
    res = cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX)
    res = res.astype(np.uint8)
    return res


def compareOutputs(diff, layer, total_img_size):

    print("#### %s ####"%(layer))
    print("Max Difference - %d"%(np.max(diff)))
    print("Min Difference - %d"%(np.min(diff)))

    num_invalid = 0
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            for k in range(diff.shape[2]):
                if abs(diff[i,j,k]) > 2:
                    num_invalid +=1
                    print("Invalid at %d, %d, %d"%(i,j,k))

    print("Error Rate - %f"%(float(num_invalid)/(total_img_size)))
    print()



#Load the image
res = loadImage('../data/sample_fox.png')


#Process the image for feeding it to the model appropriately
res = np.reshape(res, (1,res.shape[0],res.shape[1],res.shape[2]))
res = np.transpose(res, (0, 3, 1, 2))


#Create 3 models to test the 3 components
Batchsize = res.shape[0]
Channels = 3
H = res.shape[2]
W = res.shape[3]

#Model with just Conv
model_conv = nn.Sequential(
    nn.Conv2d(Channels, 3, kernel_size=(3,3), padding=1, bias=False),
)

#Model with Conv + Pool
model_pool = nn.Sequential(
    nn.Conv2d(Channels, 3, kernel_size=(3,3), padding=1, bias=False),
    nn.AvgPool2d(4),
)

#Model with Conv + Pool + Activation
model_act = nn.Sequential(
    nn.Conv2d(Channels, 3, kernel_size=(3,3), padding=1, bias=False),
    nn.AvgPool2d(4),
    nn.Sigmoid() 
)

#Model with Conv + Pool + Activation + Linear
model_lin = nn.Sequential(
    nn.Conv2d(Channels, 3, kernel_size=(3,3), padding=1, bias=False),
    nn.AvgPool2d(4),
    nn.Sigmoid(),
    nn.Flatten(),
    nn.Linear(12288,100)
)

#Create the custom Kernel
h_kernel =createKernel()

#Create weight and bias
w, b = createWeightAndBias(100,12288)

#Set the kernel to the conv layer
with torch.no_grad():
    model_conv[0].weight = nn.Parameter(torch.from_numpy(h_kernel).float())
    model_pool[0].weight = nn.Parameter(torch.from_numpy(h_kernel).float())
    model_act[0].weight = nn.Parameter(torch.from_numpy(h_kernel).float())
    model_lin[0].weight = nn.Parameter(torch.from_numpy(h_kernel).float())
    model_lin[4].weight = nn.Parameter(torch.from_numpy(w).float())
    model_lin[4].bias = nn.Parameter(torch.from_numpy(b).float())


#Create the input tensor
input_tensor = torch.from_numpy(res)

#Run the models
with torch.no_grad():
    o_conv = model_conv(input_tensor.float())
    o_pool = model_pool(input_tensor.float())
    o_act = model_act(input_tensor.float())
    o_lin = model_lin(input_tensor.float())

    out_conv = o_conv.numpy()
    out_pool = o_pool.numpy()
    out_act = o_act.numpy()
    out_lin = o_lin.numpy()


#Reshape the outputs
out_conv = np.transpose(out_conv, (0, 2, 3, 1 ))
out_conv = np.reshape(out_conv, (out_conv.shape[1], out_conv.shape[2], out_conv.shape[3]))

out_pool = np.transpose(out_pool, (0, 2, 3, 1 ))
out_pool = np.reshape(out_pool, (out_pool.shape[1], out_pool.shape[2], out_pool.shape[3]))

out_act = np.transpose(out_act, (0, 2, 3, 1 ))
out_act = np.reshape(out_act, (out_act.shape[1], out_act.shape[2], out_act.shape[3]))

#Post-process the images
out_conv = post_process(out_conv)
out_pool = post_process(out_pool)
out_act = post_process(out_act)

# showImage(out_act)



#Convolution Compare
conv_comp_img = cv2.imread('cudnnconv.png',cv2.IMREAD_COLOR)
conv_comp_img = cv2.normalize(conv_comp_img, conv_comp_img, 0, 255, cv2.NORM_MINMAX)
diff_conv = conv_comp_img - out_conv
compareOutputs(diff_conv, "Conv Layer", out_conv.size)

#Pooling Compare
pool_comp_img = cv2.imread('cudnnpool.png',cv2.IMREAD_COLOR)
pool_comp_img = cv2.normalize(pool_comp_img, pool_comp_img, 0, 255, cv2.NORM_MINMAX)
diff_pool = pool_comp_img - out_pool
compareOutputs(diff_pool, "Pool Layer", out_pool.size)

#Activation Compare
act_comp_img = cv2.imread('cudnnact.png',cv2.IMREAD_COLOR)
act_comp_img = cv2.normalize(act_comp_img, act_comp_img, 0, 255, cv2.NORM_MINMAX)
diff_act = act_comp_img - out_act
compareOutputs(diff_act, "Activation Layer", out_act.size)


#Linear Only
w, b = createWeightAndBias(10, 100)
MAX = 1024
inp = np.asarray([x%MAX for x in range(100)])
out = np.matmul(w,inp) + b
out = np.reshape(out, (1, 10))

print("#### Linear Check ####")
with open("linear.txt", "r") as f:
    lines = f.readlines()

    net_lin_out = lines[0].rsplit()
    net_lin_out = np.asarray([float(x) for x in net_lin_out])

    #Check this output - Has scope of some error
    print("Output of the Neural Net - Has Scope of error:")
    comp_net_lin = net_lin_out - out_lin
    print("Max Difference - %d"%comp_net_lin.max())
    print("Min Difference - %d"%comp_net_lin.min())
    num_invalid = 0
    for i in range(comp_net_lin.size):
        if abs(comp_net_lin[(0,i)]) > 3:
            num_invalid += 1
            print("Invalid at %d"%i)
    print("Error Rate - %f"%(float(num_invalid)/(comp_net_lin.size)))
    print()


    only_lin_out = lines[1].rsplit()
    only_lin_out = np.asarray([float(x) for x in only_lin_out])

    #Check this output - Has no scope of some error
    print("Output of the direct multiplication - Has No Scope of error:")
    comp_only_lin = only_lin_out - out
    print("Max Difference - %d"%comp_only_lin.max())
    print("Min Difference - %d"%comp_only_lin.min())
    num_invalid = 0
    for i in range(comp_only_lin.size):
        if abs(comp_only_lin[(0,i)]) > 0:
            num_invalid += 1
            print("Invalid at %d"%i)
    print("Error Rate - %f"%(float(num_invalid)/(comp_only_lin.size)))
    print()
