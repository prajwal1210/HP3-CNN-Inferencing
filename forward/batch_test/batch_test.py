import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import cv2
import sys

#Load the Image
def loadImage(imagePath):
    img = cv2.imread(imagePath,cv2.IMREAD_COLOR).astype(np.float32)
    res = cv2.resize(img, (256,256))
    res = cv2.normalize(res, res, 0.0, 1.0, cv2.NORM_MINMAX)
    return res

cpp_output = None
with open("final_out.txt", "r") as f:
    lines = f.readlines()
    cpp_output = lines[0].rstrip()
    cpp_output = cpp_output.split(' ')
    cpp_output = [float(x) for x in cpp_output]  

cpp_output = np.asarray(cpp_output)

model = None
if(sys.argv[1] == "vgg"):
    model = models.vgg19(pretrained=True)
elif(sys.argv[1] == "alex"):
    model = models.alexnet(pretrained=True)
else:
    print("Please enter the model name - \"vgg\" or \"alex\" as command line argument")
    exit(1)

#Load the image
data_path = "../data/MiniImageNet/"


with open(data_path + "ImageLists.txt", "r") as f:
    lines = f.readlines()
    image_list = lines[:8]

data = []
for img in image_list:
    img_path = data_path + img
    img_path = img_path.rstrip()
    res = loadImage(img_path)
    res = np.transpose(res, (2, 0, 1))
    data.append(res)

data = np.asarray(data)

#Create the input tensor
input_tensor = torch.from_numpy(data)

#Feed the images to the model appropriately
model.eval()
with torch.no_grad():
    output_tensor = model(input_tensor)
    output = output_tensor.numpy()

output = np.reshape(output, (-1,))

diff = cpp_output - output
diff = np.abs(diff)

print ("Average difference between two values of the output layer - %f"%(np.average(diff)))