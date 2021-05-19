import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

for i in range(4):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    for j in range(2):
        ax = plt.Subplot(fig, inner[j])
        t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (i, j))
        t.set_ha('center')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

fig.show()


import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0.1,0.5,10) # 生成[0.1,0.5]等间隔的十个数据
y=np.exp(x)
y1=y+2
error=0.05+0.15*x # 误差范围函数
error_range=[error*0.3,error] # 下置信度和上置信度

plt.errorbar(x,y,yerr=error_range,fmt='o:',ecolor='hotpink',elinewidth=3,ms=5,mfc='wheat',mec='salmon',capsize=3)
plt.errorbar(x,y1,yerr=error_range,fmt='o:',ecolor='hotpink',elinewidth=3,ms=5,mfc='wheat',mec='salmon',capsize=3)
plt.show()

























