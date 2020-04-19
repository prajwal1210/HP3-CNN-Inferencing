import os 
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('No Display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import argparse
import shutil
import seaborn as sns   
sns.set(context = "poster", font_scale = 0.95, rc={"lines.linewidth": 1.5, 'lines.markersize': 10, 'legend.frameon': True})

PLOT_DIR = "./Plots/"
ALEX_NET_LOG = "./logALEX.txt"
VGG_NET_LOG = "./logVGG.txt"



# Hard Coded #
def getAlexNetSpecification():
    # (input channels, output channels, kernel size, stride, padding)
    kernel_specifications = [(3, 64, 11, 4, 2), (64, 192, 5, 1, 2), (192, 384, 3, 1, 1), (384, 256, 3, 1, 1), (256, 256, 3, 1, 1)]
    output_specifications = [(1, 64, 63, 63), (1, 192, 31, 31), (1, 384, 15, 15), (1, 256, 15, 15), (1, 256, 15, 15)]
    return kernel_specifications, output_specifications

# Hard Coded #
def getVGGSpecification():
    kernel_specifications = [(3, 64, 3, 1, 1), (64, 64, 3, 1, 1), (64, 128, 3, 1, 1), (128, 128, 3, 1, 1), (128, 256, 3, 1, 1), (256, 256, 3, 1, 1), (256, 256, 3, 1, 1), (256, 256, 3, 1, 1), (256, 512, 3, 1, 1), (512, 512, 3, 1, 1), (512, 512, 3, 1, 1), (512, 512, 3, 1, 1), (512, 512, 3, 1, 1), (512, 512, 3, 1, 1), (512, 512, 3, 1, 1), (512, 512, 3, 1, 1)]
    output_specifications = [(1, 64, 256, 256), (1, 64, 256, 256), (1, 128, 128, 128), (1, 128, 128, 128), (1, 256, 64, 64), (1, 256, 64, 64), (1, 256, 64, 64), (1, 256, 64, 64), (1, 512, 32, 32), (1, 512, 32, 32), (1, 512, 32, 32), (1, 512, 32, 32), (1, 512, 16, 16), (1, 512, 16, 16), (1, 512, 16, 16), (1, 512, 16, 16)]
    return kernel_specifications, output_specifications

def getLayerParams(filter_spec, input_dimensions):
    number_of_Paremeters = filter_spec[0]*filter_spec[1]*(filter_spec[2]**2)
    number_of_InputValues = 1
    for d in input_dimensions:
        number_of_InputValues *= d
    return number_of_Paremeters, number_of_InputValues



def loadAlexLog():
    input_spec = [(1,3,256,256)]
    kernel_spec, output_spec = getAlexNetSpecification()
    input_spec.extend(output_spec[:-1])
        
    layer_specification_dict = {}
        
    alex_header = ["ALGO", "BATCHSIZE", "TIME_TYPE" ]
    for i,k in enumerate(kernel_spec):
        header_ = "CONV"+str(i)
        
        layer_specification_dict[header_] = {}
        layer_specification_dict[header_]['input'] = input_spec[i]
        layer_specification_dict[header_]['filter'] = k
        layer_specification_dict[header_]['output'] = output_spec[i]

        alex_header.append(header_)
    
    data = pd.read_csv(ALEX_NET_LOG, header=None, names = alex_header) 
    return alex_header, data, layer_specification_dict


def loadVGGLog():
    input_spec = [(1,3,256,256)]
    kernel_spec, output_spec = getVGGSpecification()
    input_spec.extend(output_spec[:-1])
        
    layer_specification_dict = {}
        
    vgg_header = ["ALGO", "BATCHSIZE", "TIME_TYPE" ]
    for i,k in enumerate(kernel_spec):
        header_ = "CONV"+str(i)
        
        layer_specification_dict[header_] = {}
        layer_specification_dict[header_]['input'] = input_spec[i]
        layer_specification_dict[header_]['filter'] = k
        layer_specification_dict[header_]['output'] = output_spec[i]

        vgg_header.append(header_)
    
    data = pd.read_csv(VGG_NET_LOG, header=None, names = vgg_header) 
    return vgg_header, data, layer_specification_dict

#######################
# Layer Wise Analysis #
#######################


def plotLayerWiseDataForTimeType(data, layer_cols, layer_specification_dict, batch_size, architecture, plot_path, time_type = ' CONV', label_y = 'Convolution Time (in ms)', plot_title = 'Convolution Time across Layers'):
    data_batch = data.loc[data['BATCHSIZE'] == batch_size].copy()
    data_frame_tt = data_batch.loc[data_batch['TIME_TYPE'] == time_type].copy()
    data__frame_tt_transposed = data_frame_tt.set_index('ALGO')[layer_cols].T
    data__frame_tt_transposed['Layers'] = data__frame_tt_transposed.index
    df_to_plot = data__frame_tt_transposed.melt('Layers', value_name='Time')
    
    X_ = []
    for i,row in df_to_plot.iterrows():
        num_filter_parameters, num_input_values = getLayerParams(layer_specification_dict[row['Layers']]['filter'],layer_specification_dict[row['Layers']]['input'])
        label =  row['Layers'] + "\n" + str(layer_specification_dict[row['Layers']]['filter']) + "\n" + "#P = " +str(num_filter_parameters) + "\n" + "IS = " + str(num_input_values) + "x" +str(batch_size)+"\n"
        X_.append(label)

    df_to_plot['X_LABELS'] = X_
    
    plt.figure(figsize=(max(16, len(layer_cols)*2+4),8)) # this creates a figure 8 inch wide, 4 inch high
    
    ax = sns.lineplot(x='X_LABELS', y='Time', hue='ALGO', marker = 'o', data=df_to_plot, sort=False)
    ax.set_yscale("log")
    ax.set(xlabel='Convolution Layers', ylabel = label_y)
    ax.tick_params(axis="x", labelsize = 13)
    ax.set_title(plot_title + " for Batchsize = " + str(batch_size), y=1.15)
    box = ax.get_position()
    
    ax.set_position([box.x0, box.y0, box.width , box.height * 0.95]) # resize position
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:],bbox_to_anchor=(0.5, 1.1), loc='upper center' , borderaxespad=0., ncol = len(layer_cols)-1)
    
    mean_d = df_to_plot.groupby(['X_LABELS','ALGO']).mean().reset_index()
    for i,row in mean_d.iterrows():
        x = row['X_LABELS']
        y = row['Time']
        ax.text(x,y,f'{y:.2f}\n',ha = 'center', va = 'center', clip_on=True, fontsize = 18)
    plt.savefig(plot_path+"LayerWise_"+architecture+time_type.strip()+'_'+str(batch_size)+'.png', bbox_inches='tight')
    plt.close()


def LayerWiseAnalysis(data, header, layer_specification_dict, architecture):
    layer_cols = header[3:]
    
    plot_dir_path = PLOT_DIR + architecture + "/"
    if not os.path.isdir(plot_dir_path):
        os.mkdir(plot_dir_path)
    
    time_types = [' CONV', ' OVERHEAD', ' TOTAL']
    y_labels = ['Convolution Time (in ms)', 'Overhead Time (in ms)', 'Total Time (in ms)']
    plot_titles = ['Convolution Time across Layers', 'Overhead Time across Layers', 'Total Time across Layers [Includes the extra overheads] ']
    for i,t in enumerate(time_types):
        plotLayerWiseDataForTimeType(data, layer_cols, layer_specification_dict, batch_size = 1, architecture = architecture, plot_path = plot_dir_path, time_type = t, label_y = y_labels[i], plot_title = plot_titles[i])
        plotLayerWiseDataForTimeType(data, layer_cols, layer_specification_dict, batch_size = 8, architecture = architecture, plot_path = plot_dir_path, time_type = t, label_y = y_labels[i], plot_title = plot_titles[i])




########################
# Acros Batch Analysis #
########################


def batchWisePlot(data, layer_cols, layer_specification_dict, architecture, plot_path, time_type = ' CONV', label_y = 'Sum of Convolution Time (in ms)', plot_title = 'Sum of Convolution Time Vs Batchsize'):
    data_batchwise = data.loc[data['TIME_TYPE'] == time_type].copy()
    data_batchwise['SUM'] = data_batchwise[layer_cols].sum(axis = 1)
    
    plt.figure(figsize=(14,10)) # this creates a figure 8 inch wide, 4 inch high
    ax = sns.lineplot(x='BATCHSIZE', y='SUM', hue='ALGO', marker = 'o', data=data_batchwise, sort=False)
    ax.set_yscale("log")
    ax.set(xlabel='Batch Size', ylabel = label_y)
    ax.tick_params(axis="x", labelsize = 13)
    ax.set_title(plot_title, y=1.15)
    box = ax.get_position()

    ax.set_position([box.x0, box.y0, box.width , box.height * 0.95]) # resize position

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:],bbox_to_anchor=(0.5, 1.1), loc='upper center' , borderaxespad=0., ncol = len(layer_cols)-1)

    mean_d = data_batchwise.groupby(['BATCHSIZE','ALGO']).mean()['SUM'].reset_index()
    for i,row in mean_d.iterrows():
        x = row['BATCHSIZE']
        y = row['SUM']
        ax.text(x,y,f'{y:.2f}\n',ha = 'center', va = 'center', clip_on=True, fontsize = 14)
    plt.savefig(plot_path+"BatchWise_"+architecture+time_type.strip()+'.png', bbox_inches='tight')
    plt.close()


def BatchWiseAnalysis(data, header, layer_specification_dict, architecture):
    layer_cols = header[3:]
    
    plot_dir_path = PLOT_DIR + architecture + "/"
    if not os.path.isdir(plot_dir_path):
        os.mkdir(plot_dir_path)

    time_types = [' CONV', ' OVERHEAD', ' TOTAL']
    y_labels = ['Sum of Convolution Time (in ms)', 'Sum of Overhead Time (in ms)', 'Sum of Total Time (in ms)']
    plot_titles = ['Sum of Convolution Time Vs Batchsize', 'Sum of Overhead Time Vs Batchsize', 'Sume of Total Time [Includes the extra overheads] Vs Batchsize']
    for i,t in enumerate(time_types):
        batchWisePlot(data, layer_cols, layer_specification_dict, architecture = architecture, plot_path = plot_dir_path, time_type = t, label_y = y_labels[i], plot_title = plot_titles[i])



if __name__ == "__main__":

    if not os.path.isdir(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    if not os.path.exists(VGG_NET_LOG):
        print ('logVGG.txt does not exist...Skipping analysis for VGG')
    else:
        print ('Generating plots for VGG ...')
        header, data, layer_specification_dict = loadVGGLog()
        LayerWiseAnalysis(data, header, layer_specification_dict, "VGG")
        BatchWiseAnalysis(data, header, layer_specification_dict, "VGG")

    if not os.path.exists(VGG_NET_LOG):
        print ('logALEX.txt does not exist...Skipping analysis for ALEXNET')
    else:
        print ('Generating plots for Alexnet ...')
        header, data, layer_specification_dict = loadAlexLog()
        LayerWiseAnalysis(data, header, layer_specification_dict, "ALEX")
        BatchWiseAnalysis(data, header, layer_specification_dict, "ALEX")





