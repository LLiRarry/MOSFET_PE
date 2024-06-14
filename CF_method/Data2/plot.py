import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',
    "axes.unicode_minus": False
}
rcParams.update(config)
# 加载数据
DE=np.load('DE.npy')
PSO=np.load('PSO.npy')
#EO=np.load('EO2.npy')
NNA=np.load('NNA.npy')
EO=np.load('EO2.npy')
RLEO=np.load('EO.npy')
list=[DE,PSO[0:300],NNA,EO,RLEO]
name=['DE','PSO','NNA','EO','RLEO']

#colors = plt.cm.viridis(np.linspace(0, 1, len(list)))
markers = ['v', 's', '^', 'D', 'o']
colors = plt.cm.rainbow(np.linspace(0, 1, len(list)))
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(len(list)):
    l=len(list[1])
    Advans=list[i][0:l]
    x=np.linspace(1,len(Advans),len(Advans))
    ax.scatter(x, Advans, color=colors[i], marker=markers[i], edgecolors='black', s=50,label=name[i])
    ax.plot(x, Advans, color='black', linewidth=1)
# ax.set_title('Scatter Plot with Line')
ax.set_xlabel('The number of functions evaluations')
ax.set_ylabel('Min RMSE value(log)')
# highlight_rect = plt.Rectangle((15, -1), 5, 2, fill=True,color='skyblue', alpha=0.5, linewidth=2, label='Highlight Area')
# ax.add_patch(highlight_rect)
# ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.tick_params(axis='both', which='both', direction='out', width=0.5)
plt.show()