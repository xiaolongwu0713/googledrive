'''
collect statistic about power change for each active change across different movement
'''
import sys
import time
from scipy import stats
from mne.time_frequency import tfr_morlet, tfr_multitaper
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])


from grasp.config import *
from grasp.process.channel_settings import *
from grasp.process.signalProcessUtils import getIndex


sid = 16
movements=4
# fast testing
#activeChannels[sid]=activeChannels[sid][:1]

plot_dir = data_dir + 'PF' + str(sid) + '/p_values/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

movementEpochs = []  # movementEpochs[0] is the epoch of move 1
ch_names = []
print('SID: '+str(sid) + '. Reading all 4 movement epochs.')
for movement in range(movements):
    movementEpochs.append(
        mne.read_epochs(data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch' + str(movement) + '.fif').pick(
            picks=activeChannels[sid]))
    ch_names.append(movementEpochs[movement].ch_names)
ch_names = ch_names[0]
ch_names = [str(index) + '-' + name for index, name in zip(activeChannels[6], ch_names)]

n_cycles_mthod='stage' # or: equal
fMin,fMax=2,150
fstep=1
freqs=np.arange(fMin,fMax,fstep) #148
fNum=freqs.shape[0]
#freqs = np.linspace(fMin,fMax, num=fNum)
cycleMin,cycleMax=8,50
cycleNum=fNum
#n_cycles = np.linspace(cycleMin,cycleMax, num=cycleNum)  # different number of cycle per frequency
groups=5
rates=[2,2.5,3,4,5]
num_per_group=int(fNum/groups)
if n_cycles_mthod=='equal':
    n_cycles=freqs
elif n_cycles_mthod=='stage':
    n_cycles=[]
    for g in range(groups):
        if g < groups -1:
            tmp=[int(i) for i in freqs[g*num_per_group:(g+1)*num_per_group]/rates[g]]
        elif g==groups -1:
            tmp = [int(i) for i in freqs[g * num_per_group:] / rates[g]]
        n_cycles.extend(tmp)


crop1 = 0
crop2 = 15
decim = 4
new_fs = 1000 / decim
base1 = 10  # s
base2 = 13  # s
erds_span = [[1.5, 8.0], [1.5, 14.0], [1.5, 6.0], [1.5, 8.0]]
baseline = [[] for _ in range(movements)]
baseline[0] = [int((10 - crop1) * new_fs), int((13 - crop1) * new_fs)]
baseline[1] = [int((14 - crop1) * new_fs), int((15 - crop1) * new_fs)]
baseline[2] = [int((10 - crop1) * new_fs), int((13 - crop1) * new_fs)]
baseline[3] = [int((10 - crop1) * new_fs), int((13 - crop1) * new_fs)]

movementsLines=[[0,2,5,7.5,15],[0,2,11,13.5,15],[0,2,3,5.5,15],[0,2,5,7.5,15]]
task_durations=[]
for i in range(len(movementsLines)):
    task_durations.append([])
    task_durations[i]=[int(new_fs*movementsLines[i][1]),int(new_fs*movementsLines[i][3])]


sub_bands_number = 2
erd_wind = 10
ers_wind = 2
erd_end_f = 30
ers_start_f = 30
normalization = 'db'  # 'z-score'/'db'

erd_change = []  # ers_change[movement][channel][trials....]
ers_change = []
for chIndex, chName in enumerate(ch_names):
    print('Computing TF on ' + str(chIndex) + '/' + str(len(ch_names)) + ' channel.')
    # print('Processing channel ' + chName + '.')
    erd_change.append([])
    ers_change.append([])
    for movement in range(movements):
        erd_change[chIndex].append([])
        ers_change[chIndex].append([])
        # one_channel=movementEpochs[movement].copy().pick(picks=[chIndex]) # pick the channle below
        one_channel_tf = np.squeeze(tfr_morlet(
            movementEpochs[movement], picks=[chIndex], freqs=freqs, n_cycles=n_cycles, use_fft=True,
            return_itc=False, average=False, decim=decim, n_jobs=1).data)
        # (40, 148, 3751)
        # ERS/ERD of all trials
        for trial in range(40):
            # erd_change[movement][chIndex].append([])
            # ers_change[movement][chIndex].append([])
            base = one_channel_tf[trial, :, baseline[movement][0]:baseline[movement][1]]
            basemean = np.mean(base, 1)
            basestd = np.std(base, 1)

            if normalization == 'z-score':
                # Method:z-score
                one_channel_tf[trial] = one_channel_tf[trial] - basemean[:, None]
                one_channel_tf[trial] = one_channel_tf[trial] / basestd[:, None]
            elif normalization == 'db':
                # Method:db
                one_channel_tf[trial] = 10 * np.log10(one_channel_tf[trial] / basemean[:, None])

            mean_change = np.mean(one_channel_tf[trial][:, task_durations[movement][0]:task_durations[movement][1]], axis=1)

            erd_start_f_index = getIndex(fMin, fMax, fstep, fMin)
            erd_end_f_index = getIndex(fMin, fMax, fstep, erd_end_f)  # 30

            ers_start_f_index = getIndex(fMin, fMax, fstep, ers_start_f)  # 50
            ers_end_f_index = getIndex(fMin, fMax, fstep, fMax)

            # moving window average
            erd_wind_avg = np.convolve(mean_change[erd_start_f_index:erd_end_f_index], np.ones(erd_wind) / erd_wind,
                                       mode='valid')
            ers_wind_avg = np.convolve(mean_change[ers_start_f_index:ers_end_f_index], np.ones(ers_wind) / ers_wind,
                                       mode='valid')
            # find the max active frequency
            erd_f1 = erd_wind_avg.argmin(axis=0)
            erd_f2 = erd_f1 + erd_wind
            ers_f1 = ers_start_f_index + ers_wind_avg.argmax(axis=0)
            ers_f2 = ers_start_f_index + ers_f1 + ers_wind

            erd = np.mean(one_channel_tf[trial][erd_f1:erd_f2, :], 0)  # It's a line.
            ers = np.mean(one_channel_tf[trial][ers_f1:ers_f2, :], 0)

            erd_compare_with = np.mean(erd[baseline[movement][0]:baseline[movement][1]])
            ers_compare_with = np.mean(ers[baseline[movement][0]:baseline[movement][1]])

            erd_change[chIndex][movement].append(
                (np.min(erd[task_durations[movement][0]:task_durations[movement][1]])) - erd_compare_with)
            ers_change[chIndex][movement].append(
                (np.max(ers[task_durations[movement][0]:task_durations[movement][1]])) - ers_compare_with)

            #erd_change[chIndex][movement].append(np.mean(one_channel_tf[erd_f1:erd_f2,task_durations[movement][0]:task_durations[movement][1]]))
            #ers_change[chIndex][movement].append(np.mean(one_channel_tf[ers_f1:ers_f2,task_durations[movement][0]:task_durations[movement][1]]))

def label_diff(ax,i, j, data, X, Y, Y_std):
    if type(data) is str:
        text = data
    else:
        text = ''
        p = .05
        while data < p:
            text += '*'
            p /= 10.
        if len(text) == 0:
            text = 'n.s.'

    annotations = [child for child in ax.get_children() if isinstance(child, matplotlib.text.Annotation)]
    y_values = [ann.get_position()[1] for ann in annotations if float(ann.get_position()[0])>=float(i) and float(ann.get_position()[0])<=float(j) ]

    x = (X[i] + X[j]) / 2
    #props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 5, 'shrinkB': 5, 'linewidth': 2}
    props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'linewidth': 2}
    if Y[i]<0 and Y[j]<0:
        y = 1.1 * min(Y[i]-Y_std[i], Y[j]-Y_std[j])
        y_values.append(y)
        y = min(y_values)
        ax.annotate('', xy=(X[j], y), xytext=(X[i], y), arrowprops=props, ha='center', va='bottom')
        ax.annotate(text, xy=(x-0.2, y-3), zorder=10)
    elif Y[i]>0 and Y[j]>0:
        y = 1.1 * max(Y[i]+Y_std[i], Y[j]+Y_std[j])
        y_values.append(y)
        y = max(y_values)
        ax.annotate('', xy=(X[i], y), xytext=(X[j], y), arrowprops=props, ha='center', va='bottom')
        ax.annotate(text, xy=(x-0.2, y + 1), zorder=10)

def barplot_annotate_brackets(layer,num1, num2, data, center, height, yerr=None, dh=.05, barh=.02, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.
    :param layer: 2nd layer bracket need on top on first layer bracket.
    :param num1: 1st bar
    :param num2: 2nd bar
    :param data: p-value or other string
    :param center: bar location
    :param height: mean
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: relative position of braket tips
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        if ly<0 and ry<0:
            ly -= yerr[num1]
            ry -= yerr[num2]
        elif ly>0 and ry>0:
            ly += yerr[num1]
            ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    barx = [lx, lx, rx, rx]
    if ly < 0 and ry < 0:
        y = min(ly, ry) - dh
        if layer!=0:
            bary = [y, y - barh, y - barh, y]
            bary = [i - barh * layer - 2*barh for i in bary]
        elif layer==0:
            bary = [y, y - barh, y - barh, y]
        text_y = min(bary)
        # bary = [i - barh * layer - 0.5 for i in bary]
        mid = ((lx + rx) / 2, text_y - 2 * barh)
    elif ly > 0 and ry > 0:
        y = max(ly, ry) + dh
        if layer!=0:
            bary = [y, y + barh, y + barh, y]
            bary=[i + barh * layer + 2*barh for i in bary]
        elif layer==0:
            bary = [y, y + barh, y + barh, y]
        text_y = max(bary)
        #bary = [i + barh * layer + 0.5 for i in bary]
        mid = ((lx + rx) / 2, text_y)
    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

# power change line plot
fig, ax = plt.subplots()
print('Plotting ERDS statistics...')
for channel in range(len(ch_names)):
    ax.clear()
    erd_move0 = erd_change[channel][0] # 40 values
    erd_move1 = erd_change[channel][1]
    erd_move2 = erd_change[channel][2]
    erd_move3 = erd_change[channel][3]

    ers_move0 = ers_change[channel][0]
    ers_move1 = ers_change[channel][1]
    ers_move2 = ers_change[channel][2]
    ers_move3 = ers_change[channel][3]

    erd_datasets = [erd_move0, erd_move1, erd_move2, erd_move3]
    ers_datasets = [ers_move0, ers_move1, ers_move2, ers_move3]

    x = np.array([1, 2, 3, 4])  #
    xlabel = ['20% MVC slow', '26% MVC slow', '20% MVC fast', '60% MVC fast', ]
    erd_mean = [np.mean(dataset) for dataset in erd_datasets] # 4 values
    ers_mean = [np.mean(dataset) for dataset in ers_datasets]
    erd_std = [np.std(dataset) for dataset in erd_datasets]
    ers_std = [np.std(dataset) for dataset in ers_datasets]

    #ax.errorbar(x=x, y=erd_mean, yerr=erd_std, fmt='-o', ecolor='orange', elinewidth=1, ms=5, mfc='wheat', mec='salmon',capsize=3)
    ax.bar(x, erd_mean, yerr=erd_std, color=['brown'],
           error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=2))
    ax.bar(x+5, ers_mean, yerr=ers_std,color=['orange'],
           error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=2))
    ax.legend(['ERD', 'ERS'],loc="lower left", bbox_to_anchor=(0, 0.8))
    ax.axhline(y=0, color='r', linestyle='--')
    for i in range(movements): # simulate this horizon line as x-axis.
        ax.text(i + 0.9, 1, 'M' + str(i + 1), fontsize=8) # xticklabels above
        ax.text(i + 0.9+5, -1.6, 'M' + str(i + 1), fontsize=8) # xticklabels below
    ax.set_xticks([])
    #fontdict = {'fontsize': 8}
    ax.set_xticklabels([])
    ax.set_ylabel('Change %')
    # plt.show()
    # save
    #figname = plot_dir + 'ERSD_stat_change' + str(channel) + '.png'
    #fig.savefig(figname, dpi=400)
    #ax.clear()
    # power change error bar plot with p-value(single-tailed paired t-test)
    # Pull the formatting out here
    #bar_kwargs = {'width': 0.5, 'linewidth': 2, 'zorder': 5}
    #err_kwargs = {'zorder': 0, 'fmt': 'none', 'linewidth': 2,'ecolor': 'k'}  # for matplotlib >= v1.4 use 'fmt':'none' instead
    # plot bar
    #ax.p1 = plt.bar(x, erd_mean, color=['red', 'green', 'blue', 'cyan'],**bar_kwargs)
    #ax.errs1 = plt.errorbar(x, erd_mean, yerr=erd_std, **err_kwargs)
    #ax.bar(x, erd_mean, yerr=erd_std, color=['red', 'green', 'blue', 'cyan'],error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=2))
    #ax.p2 = plt.bar(x + 5, ers_mean, **bar_kwargs)
    #ax.errs2 = plt.errorbar(x + 5, ers_mean, yerr=ers_std, **err_kwargs)
    #ax.axhline(y=0, color='r', linestyle='--')

    ylim1, ylim2 = ax.get_ylim()
    ax.set_ylim(ylim1 - 10, ylim2 + 10)

    _, p01_erd = stats.ttest_rel(erd_move0, erd_move1) #, alternative='less')
    _, p23_erd = stats.ttest_rel(erd_move2, erd_move3)#, alternative='less')
    _, p13_erd = stats.ttest_rel(erd_move1, erd_move3)#, alternative='less')

    barplot_annotate_brackets(0,0, 1, p01_erd, x, erd_mean, erd_std)
    barplot_annotate_brackets(0,2, 3, p23_erd, x, erd_mean, erd_std)
    barplot_annotate_brackets(1,1, 3, p13_erd, x, erd_mean, erd_std)

    #label_diff(ax, 0, 1, p01_erd, x, erd_mean, erd_std)
    #label_diff(ax, 2, 3, p23_erd, x, erd_mean, erd_std)
    #label_diff(ax, 1, 3, p13_erd, x, erd_mean, erd_std)

    _, p01_ers = stats.ttest_rel(ers_move0, ers_move1)
    _, p23_ers = stats.ttest_rel(ers_move2, ers_move3)
    _, p13_ers = stats.ttest_rel(ers_move1, ers_move3)
    barplot_annotate_brackets(0,0, 1, p01_ers, x+5, ers_mean, ers_std)
    barplot_annotate_brackets(0,2, 3, p23_ers, x+5, ers_mean, ers_std)
    barplot_annotate_brackets(1,1, 3, p13_ers, x+5, ers_mean, ers_std)
    #label_diff(ax, 0, 1, p01_ers, x+5, ers_mean, ers_std)
    #label_diff(ax, 2, 3, p23_ers, x+5, ers_mean, ers_std)
    #label_diff(ax, 1, 3, p13_ers, x+5, ers_mean, ers_std)

    ylim1, ylim2 = ax.get_ylim()
    ax.set_ylim(ylim1 - 3, ylim2 + 3)

    figname = plot_dir + 'p_values' + str(channel) + '.pdf'
    fig.savefig(figname, dpi=400)
    print('Plot to '+plot_dir)
    #plt.pause(0.2)


