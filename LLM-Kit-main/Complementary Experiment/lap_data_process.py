import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from fractions import Fraction
from matplotlib.patches import Rectangle
#dataset
train_file = 'processed\laptop_train.npy'
pred_file =  "lap_pre.npy"


print('Loading dataset......')
data = np.load(train_file,allow_pickle=True)
#dev_data = np.load(dev_file,allow_pickle=True)
#assert(train_data.shape[1]%3==1)
#assert(dev_data.shape[1]%3==1)
tru_label = data[:,1:]
st = data[:,0]
pred = np.load(pred_file,allow_pickle=True)

Err_st = np.zeros(len(st),dtype=object)
Err_asp = np.zeros(10,dtype=object)

for i in range(len(st)):
    for j in range(10):
        if pred[i][j] != tru_label[i][j]:
            Err_st[i] += 1
            Err_asp[j] += 1

print(Err_asp)

tru_st = 0
for i in range(len(st)):
    if Err_st[i] == 0:
        tru_st +=1
print("ACC:",(tru_st)/len(st))

Err = 0
for i in range(len(st)):
    if Err_st[i] > 0 :
        Err +=1
print("Accuracy:",(len(st)-Err)/len(st))
Acc_asp =[]
for i in range(10):
    Acc_asp.append((len(st)-Err_asp[i])/len(st))
print(Acc_asp)

aspect = ["LAPTOP#GENERAL", "LAPTOP#OPERATION_PERFORMANCE", "LAPTOP#DESIGN_FEATURES", "LAPTOP#QUALITY", "LAPTOP#MISCELLANEOUS", 
          "LAPTOP#USABILITY", "SUPPORT#QUALITY", "LAPTOP#PRICE", "COMPANY#GENERAL", "BATTERY#OPERATION_PERFORMANCE" ]

asp_4 = []
asp_3 = []
for i in range(len(Err_st)):
    if Err_st[i] == 5:
        asp_4.append(i)
    elif Err_st[i] == 3:
        asp_3.append(i)

for id in asp_4:
    print("Sentence:",st[id],'\n',"pred:",pred[id],"\nTruth:",tru_label[id],'\n')
x = np.arange(10)

fig , ax = plt.subplots()


def to_percent(x, pos):
    return f'{x * 100:.0f}%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))

bars = ax.bar(x,Acc_asp, label='Accuracy')
#ax.set_title('Accuracy Of Different Aspect Categories ')
#ax.set_xlabel('Aspect Category')
ax.spines['top'].set_visible(False)
ax.set_ylabel('Accuracy')
ax.set_xticks(range(len(aspect)))
ax.set_xticklabels(aspect,rotation=45, ha='right', va='bottom',fontsize=6)
ax.legend()
plt.tight_layout()

plt.savefig("laptop.pdf")