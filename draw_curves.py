import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import argparse

parser_obj = argparse.ArgumentParser()

parser_obj.add_argument('--inputs', type=str, default='outputs/measures', help='path to folder containing measures')
parser_obj.add_argument('--outputs', type=str, default='curves', help='path to store adversarial images')
parser_obj.add_argument('--upper', type=int, default=1, required=False, help="upper limit of distortion to plot")
parser_obj = parser_obj.parse_args()
folder = parser_obj.inputs
folder_name = folder[:-folder[::-1].index('/')-1]
upper_limit =parser_obj.upper
print(folder, upper_limit)

def make_data(np_array, up_lim):
    res_array = np.zeros((2, np_array.shape[0]+1))
    np_array.sort()
    cpt = 0
    for j in range(np_array.shape[0]):
        res_array[1,j] = 100*cpt/np_array.shape[0]
        res_array[0,j] = np_array[j]
        cpt+=1
    res_array[1,-1] = res_array[1,-2]
    res_array[0,-1] = up_lim
    return(res_array)

colors = ['b--', 'g--','r--', 'c--','k', 'y', 'k--', 'c', 'c--','b', 'g','r', 'c','k', 'y', 'g--', 'c', 'c--']
fig_name = '{}/{}'.format(parser_obj.outputs, folder_name)
upper_limit = 1

if False:
    plt.rc('text', usetex=True)
    os.environ["PATH"] += os.pathsep + '/usr/bin/latex'
    pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
    plt.rcParams.update(pgf_with_rc_fonts)
    plt.rc('font', family='serif')
fig, axs = plt.subplots(figsize=(8,6))
axs.set_xlim([0,upper_limit])

files = glob.glob('{}/*npy'.format(folder))
print("plotting: ",files)
files.sort()

for cpt, file_id in enumerate(files):
    label_id = ' '.join(file_id.split('_'))
    file_array = np.load(file_id)
    data_id  = make_data(file_array, upper_limit)
    curve = axs.plot(data_id[0,:], data_id[1,:], colors[cpt], label=label_id, linewidth=2)
    xvalues = curve[0].get_xdata()
    yvalues = curve[0].get_ydata()
    cpt+=1

fig.legend(shadow=True, loc=(0.4, 0.2), handlelength=1.5)
plt.xlabel("distortion", {'fontsize': 20})
plt.ylabel("success rate (%)", {'fontsize': 20})
plt.grid( linestyle="dashed")
plt.savefig(fig_name)
