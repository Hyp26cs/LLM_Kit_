import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from fractions import Fraction


pred_path = "res_pre.npy"
tru_path = "processed\res_train.npy"

print("load dataset:\n")
pred = np.load(pred_path,allow_pickle=True)
tru = np.load(tru_path,allow_pickle=True)
tru_label = tru[:,1:]
st =tru[:,0]
aspect_num = tru_label.shape[1]
#print(aspect_num)

Err_st = np.zeros(len(pred),dtype=object)
Err_asp = np.zeros(aspect_num,dtype=object)
#print(Err_asp)


for i in range(len(pred)):
    for j in range(aspect_num):
        if(pred[i][j]!=tru_label[i][j]):
            Err_st[i] +=1    
            Err_asp[j] +=1  

print(Err_asp)

Err = 0
for i in range(len(tru_label)):
    if Err_st[i] > 0:
        Err += 1
print("Acc:",(len(tru_label)-Err)/len(tru_label))
idx_4 = []
idx_3 = []
for i in range(len(Err_st)):
    if Err_st[i]==4:
        idx_4.append(i)
    elif Err_st[i] ==3:
        idx_3.append(i)
# for i in range(len(idx_4)):
#     id = idx_4[i]
#     print(st[id],'\n','pred:',pred[id],'\n',"trut:",tru_label[id])

lst =[]
for i in range(aspect_num):
    lst.append((len(pred)-Err_asp[i])/(len(pred)))
print(lst)
aspect = [{"FOOD#QUALITY", "FOOD#STYLE_OPTIONS", "RESTAURANT#GENERAL", "SERVICE#GENERAL", "AMBIENCE#GENERAL",
        "DRINKS#STYLE_OPTIONS", "FOOD#PRICES", "RESTAURANT#PRICES", "LOCATION#GENERAL" , "DRINKS#QUALITY", "RESTAURANT#MISCELLANEOUS", "DRINKS#PRICES"}]

x = np.arange(aspect_num)

fig , ax = plt.subplots()


def to_percent(x, pos):
    return f'{x * 100:.0f}%'

ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))

bars = ax.bar(x,lst, label='Accuracy')
ax.set_title('Accuracy Of Different Aspect Categories ')
ax.set_xlabel('Aspect Category')
ax.set_ylabel('Accuracy')
ax.set_xticks(range(len(aspect)))
ax.set_xticklabels(aspect,ha='right')
ax.legend()
plt.tight_layout()

fig.savefig("/KGRL/KGRL-main/plot.png")



