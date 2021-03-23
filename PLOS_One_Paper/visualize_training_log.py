import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
sns.set_palette('pastel')

import pandas as pd
df=pd.read_csv('train_6LRes_log5.csv', sep=',')
print(list(df))
# df = df.drop(df.index[range(288,315)])

"""
Create plots of the training process for one epoch, showing Loss and Accuracy 
in training and validation epochs. The same is done for the denovo network
and the Transfer learning setup
"""

plt.figure()
ax1 = plt.subplot(111)
ax1.set_ylim(5, 80)
# ax1.set_yticks([15, 30, 45, 60, 75, 90, 105])
lab1=plt.plot(range(0,len(df)), df.loc[:,'loss'], color="#6094DB", label='Training - Loss')#, 'xkcd:ocean blue', label='Training - Loss',linewidth=2)
lab4=plt.plot(range(0,len(df)), df.loc[:,'val_loss'], color="#FFBD82", label='Validation - Loss')
ax1.set_xlabel('Epochs', fontsize=18, labelpad=10)
ax1.set_ylabel('Loss', fontsize=18, labelpad=10)


ax2 = ax1.twinx()
ax2.set_ylim(0, 1)
ax2.set_ylabel('Metric = Accuracy', fontsize=18, labelpad=10)
lab2=ax2.plot(range(0,len(df)), df.loc[:,'acc'], color="#44B4D5", label='Training - Accuracy', linewidth=2)#, 'xkcd:orange', label='Training - AUC',linewidth=2, linestyle='-')
lab3=ax2.plot(range(0,len(df)), df.loc[:,'val_acc'], color='#FF9797',label='Validation - Accuracy', linewidth=2)#, 'xkcd:orange', label='Validation - AUC',linewidth=2,linestyle='-.')# plt.gri
plt.grid()
lns = lab1+ lab4+ lab2+ lab3
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=5, fontsize=14)
plt.grid()
plt.savefig('training_cnn_set5.pdf', bbox_inches='tight')
# plt.show()


df1=pd.read_csv('train_TL_log5.csv', sep=',')
plt.figure(2)
ax1 = plt.subplot(111)
ax1.set_ylim(5, 80)
# ax1.set_yticks([15, 30, 45, 60, 75, 90, 105])
lab1=plt.plot(range(0,len(df1)), df1.loc[:,'loss'], color="#6094DB", label='Training - Loss')#, 'xkcd:ocean blue', label='Training - Loss',linewidth=2)
lab4=plt.plot(range(0,len(df1)), df1.loc[:,'val_loss'], color="#FFBD82", label='Validation - Loss')
ax1.set_xlabel('Epochs', fontsize=18, labelpad=10)
ax1.set_ylabel('Loss', fontsize=18, labelpad=10)

ax2 = ax1.twinx()
ax2.set_ylim(0, 1)
ax2.set_ylabel('Metric = Accuracy', fontsize=18, labelpad=10)
lab2=ax2.plot(range(0,len(df1)), df1.loc[:,'acc'], color="#44B4D5", label='Training - Accuracy', linewidth=2)#, 'xkcd:orange', label='Training - AUC',linewidth=2, linestyle='-')
lab3=ax2.plot(range(0,len(df1)), df1.loc[:,'val_acc'], color='#FF9797',label='Validation - Accuracy', linewidth=2)#, 'xkcd:orange', label='Validation - AUC',linewidth=2,linestyle='-.')# plt.gri
plt.grid()
lns = lab1+ lab4+ lab2+ lab3
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=5, fontsize=14)
plt.grid()
plt.savefig('training_TL_set5.pdf', bbox_inches='tight')
plt.show()

