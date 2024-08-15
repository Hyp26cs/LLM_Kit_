import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

labels = ["LAPTOP#GENERAL", "LAPTOP#OPERATION_PERFORMANCE", "LAPTOP#DESIGN_FEATURES", "LAPTOP#QUALITY", "LAPTOP#MISCELLANEOUS", 
          "LAPTOP#USABILITY", "SUPPORT#QUALITY", "LAPTOP#PRICE", "COMPANY#GENERAL", "BATTERY#OPERATION_PERFORMANCE" ]
accuracy = [40.307692307692305, 74.15384615384616, 82.76923076923077, 79.84615384615384, 82.46153846153846, 
            84.76923076923077, 91.38461538461539, 92.15384615384615, 86.76923076923077, 98.76923076923076]

x = np.arange(len(labels))  


fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.35


bars = ax.bar(x - bar_width/2, accuracy, width=bar_width, label='Accuracy', color='skyblue', hatch='//')

ax.spines['top'].set_visible(False)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=5,ha='right',color='w')


#ax.set_ylabel('%')

plt.ylabel('%',
           labelpad=2,  
           y=1.02,  
           rotation=0,fontsize=9)
ax.set_ylim(30,100)
ax.tick_params(bottom=False, top=False, left=True, right=False)
plt.rcParams['ytick.direction'] = 'out'

ax.legend()

plt.bar_label(bars,labels=labels,label_type="edge",fontsize=5,rotation=45)
plt.tight_layout()
#plt.box(on=True)
plt.savefig("laptop.pdf")