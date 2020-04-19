import os 
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import argparse
import shutil
import seaborn as sns   
sns.set(context = "poster", font_scale = 0.95, rc={"lines.linewidth": 1.5, 'lines.markersize': 10, 'legend.frameon': True})


MEM_FILE = {}
COLORS = ['deepskyblue','green','darkred','gold','violet']
PLOT_DIR = "./Plots/"


def plotGPUData(mem_file, colors, plot_dir):
    plt.figure(figsize=(20,8))
    for i,k in enumerate(mem_file.keys()):
        Y = []
        with open(mem_file[k],'r') as f:
            csv_r = csv.reader(f)
            head = next(csv_r)
            for row in csv_r:
                val_str = row[0]
                val_str = val_str.split(' ')
                val = float(val_str[0])
                Y.append(val)
        Y = np.array(Y)
        Y = Y - Y[0]
        X = list(range(Y.size))

        plt.fill_between( X, Y, color=colors[i], alpha=0.2)
        plt.plot(X, Y, color=colors[i], alpha=0.8, label = k)

    plt.title("GPU Memory Used for one VGG19 Run [Batchsize = 1]")
    plt.xlabel('Sample Time Intervals [1 Time Interval = 100 ms]' + r'$\rightarrow$')
    plt.ylabel('Used GPU Memory (in MB)' + r'$\rightarrow$')
    plt.legend()
    plt.savefig(plot_dir+"GPUMemoryUtilization.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    
    if not os.path.isdir(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    #Direct Convolution File
    if not os.path.exists("./memory_direct.txt"):
        print ("Direct Convolution log not found..Skipping")
    else:
        MEM_FILE['DIRECT'] = "./memory_direct.txt"
    
    #Im2Col Convolution File
    if not os.path.exists("./memory_im2col.txt"):
        print ("Im2Col Convolution log not found..Skipping")
    else:
        MEM_FILE['IM2COL'] = "./memory_im2col.txt"
    
    #FFT Convolution File
    if not os.path.exists("./memory_fft.txt"):
        print ("FFT Convolution log not found..Skipping")
    else:
        MEM_FILE['FFT'] = "./memory_fft.txt"
    
    #Winograd Convolution File
    if not os.path.exists("./memory_winograd.txt"):
        print ("Winograd Convolution log not found..Skipping")
    else:
        MEM_FILE['WINOGRAD'] = "./memory_winograd.txt"
    
    #CUDNN Convolution File
    if not os.path.exists("./memory_cudnn.txt"):
        print ("CUDNN Convolution log not found..Skipping")
    else:
        MEM_FILE['CUDNN'] = "./memory_cudnn.txt"
        
    plotGPUData(MEM_FILE, COLORS, PLOT_DIR)

