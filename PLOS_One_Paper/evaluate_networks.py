import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import sklearn.metrics as skm
plt.style.use('seaborn')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
from sklearn.metrics import confusion_matrix

"""4Layer Small network, downsampled  """
filepath1 = "/Users/bhavnaantony/Code/research-bhavna/oct_key_frame/PLOS_Paper_Results/CAM6LRes_testSet6.csv"
""" TL network"""
filepath2 = "/Users/bhavnaantony/Code/research-bhavna/oct_key_frame/PLOS_Paper_Results/TL_testSet6.csv"
""" original set"""
filepath3 = "/Users/bhavnaantony/Code/research-bhavna/oct_key_frame/Duke_keyframe_table_setlabels.csv"


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap='GnBu')
    # plt.matshow(cm, cmap='GnBu')
    ttx=plt.title(title, size=20)
    ttx.set_position([0.5, 1.05])
    cbar=plt.colorbar(drawedges=True)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.25)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=16)
    plt.yticks(tick_marks, classes, size=16)
    plt.grid('off')
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),size=18,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)

if __name__ == "__main__":

    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)

    print(list(df1))
    print(list(df2))

    t = 0.55

    class_names = ['Normal', 'Abnormal']
    cnf = confusion_matrix(df1.loc[:, 'Triage_Score'], df1.loc[:, 'Pred_Score']>=t)
    print(cnf)
    plt.figure(1)
    plot_confusion_matrix(cnf, class_names, True, "Confusion Matrix: De Novo Network")
    plt.savefig('CNN_ConfMat_set6.pdf',bbox_inches='tight')
    # plt.show()

    cnf2 = confusion_matrix(df2.loc[:, 'Triage_Score'], df2.loc[:, 'Pred_Score'] > 0.5)
    plt.figure(2)
    plot_confusion_matrix(cnf2, class_names, True, "Confusion Matrix: Transfer Learning")
    plt.savefig('TL_ConfMat_set6.pdf',bbox_inches='tight')
    plt.show()

    """ plot results"""
    true_val1 = df1.loc[:, 'Triage_Score']
    pred_val1 = df1.loc[:, 'Pred_Score']
    true_val2 = df2.loc[:, 'Triage_Score']
    pred_val2 = df2.loc[:, 'Pred_Score']

    auc1 = skm.roc_auc_score(true_val1, pred_val1)
    auc2 = skm.roc_auc_score(true_val2, pred_val2)
    print("CNN AUC = ", auc1, " TL AUC = ", auc2)
    fpr1 = dict()
    tpr1 = dict()
    fpr2 = dict()
    tpr2 = dict()

    fpr1[1], tpr1[1], t = skm.roc_curve(true_val1, pred_val1)
    fpr2[1], tpr2[1], t = skm.roc_curve(true_val2, pred_val2)
    ax1 = plt.subplot(111)
    # ax1.set_facecolor('xkcd:light grey')
    plt.xlabel('1-Specificity', size=18)
    plt.ylabel('Sensitivity', size=18)
    plt.title('AUC Curves', size=20)
    plt.xticks(size=16)
    plt.yticks(size=16)
    l1 = plt.plot(fpr1[1], tpr1[1], label='de novo: {:0.2f}'.format(auc1))
    l2 = plt.plot(fpr2[1], tpr2[1], label='TL: {:0.2f}'.format(auc2))
    plt.legend(loc=5, fontsize=18)
    plt.grid(linestyle='-.',axis='both')
    plt.savefig('AUC_set6.pdf', bbox_inches='tight')
    # plt.show()
