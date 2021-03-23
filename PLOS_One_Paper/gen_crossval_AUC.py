import numpy as np

from sklearn.metrics import roc_curve, auc
from scipy.stats import ttest_ind, wilcoxon
import pandas as pd
import matplotlib.pyplot as plt


def get_kfold_auc(fname_substr, num_folds):

    rauc_TL = np.zeros(shape=(num_folds,))
    for k in range(num_folds):
        df=pd.read_csv(fname_substr + str(k) + ".csv", sep=',')
        # print(df.shape, list(df))

        y_true = np.array(df.loc[:, 'Triage_Score'])
        y_pred = np.array(df.loc[:, 'Pred_Score'])

        fpr = dict()
        tpr = dict()

        # fpr[0], tpr[0], t = roc_curve(1-y_true, 1-y_pred)
        fpr[1], tpr[1], t = roc_curve(y_true, y_pred)
        rauc_TL[k] = auc(fpr[1], tpr[1])
        print(auc(fpr[1], tpr[1]))
    return rauc_TL


if __name__ == "__main__":
    """ variables to contain results"""
    fpr_tl = np.zeros(shape=(10,))
    fnr_tl = np.zeros(shape=(10,))
    fpr_tlc = np.zeros(shape=(10,)) #Control
    fnr_tlm = np.zeros(shape=(10,)) #Mild

    fpr_dn = np.zeros(shape=(10,))
    fnr_dn = np.zeros(shape=(10,))
    fpr_dnc = np.zeros(shape=(10,)) #Controls
    fnr_dnm = np.zeros(shape=(10,)) #mild

    for k in range(10):
        print("SPLIT: ", k)
        ftl = pd.read_csv("TL_testSet" + str(k) + ".csv")
        fdn = pd.read_csv("CAM6LRes_testSet" + str(k) + ".csv")
        o = pd.read_csv("/Users/bhavnaantony/Code/research-bhavna/oct_key_frame/Duke_keyframe_table_setlabels.csv")
        # print(list(fdn))
        # print(list(ftl))

        ftl['dn_Pred'] = pd.Series(index=ftl.index)
        ftl['Orig_Score'] = pd.Series(index=ftl.index)

        for i,r in ftl.iterrows():
            # print(r['Filename'])
            # print((fdn.loc[(fdn['Filename'] == r['Filename']) & (fdn['Slice_Number'] == r['Slice_Number']), 'Pred_Score']))
            ftl.loc[i,'dn_Pred'] = fdn.loc[(fdn['Filename'] == r['Filename']) & (fdn['Slice_Number'] == r['Slice_Number']),'Pred_Score'].values
            ftl.loc[i, 'Orig_Score'] = o.loc[(o['Filename'] == r['Filename']) & (o['Slice_Number'] == r['Slice_Number']), 'Triage_Score'].values

        ftl = ftl.drop(columns=['Unnamed: 0', 'split_label'])
            # print(ftl.head(10))
        fp_tl = ftl.loc[(ftl['Triage_Score'] == 0) & (ftl['Pred_Score'] > 0.5) & (~ftl['Filename'].str.contains('1114_ctr_vol')),:]
        fp_dn = ftl.loc[(ftl['Triage_Score'] == 0) & (ftl['dn_Pred'] > 0.5) & (~ftl['Filename'].str.contains('1114_ctr_vol')),:]
        # print(fp_tl.loc[:, 'Filename'].unique())
        fpr_tl[k] = len(fp_tl)/len(ftl)
        fpr_dn[k] = len(fp_dn) / len(ftl)
        fpr_tlc[k] = len(fp_tl.loc[fp_tl['Filename'].str.contains('ctr')])/len(fp_tl)
        fpr_dnc[k] = len(fp_dn.loc[fp_dn['Filename'].str.contains('ctr')]) / len(fp_tl)

        fn_tl = ftl.loc[(ftl['Triage_Score'] == 1) & (ftl['Pred_Score'] <= 0.5), :]
        fn_dn = ftl.loc[(ftl['Triage_Score'] == 1) & (ftl['dn_Pred'] <= 0.5), :]

        fnr_tl[k] = len(fn_tl)/len(ftl)
        fnr_dn[k] = len(fn_dn) / len(ftl)
        fnr_tlm[k] = len(fn_tl.loc[fn_tl['Orig_Score'].str.contains('Mild')])/len(fn_tl)
        fnr_dnm[k] = len(fn_dn.loc[fn_dn['Orig_Score'].str.contains('Mild')]) / len(fn_dn)


        print("FP TL:", len(fp_tl), " FP DN:", len(fp_dn))
        print("Common FP=", len(pd.merge(fp_dn, fp_tl, how='inner', on=['Filename', 'Slice_Number'])))
        print("FP TL CTR", len(ftl.loc[(ftl['Triage_Score'] == 0) & (ftl['Pred_Score'] > 0.5) & (ftl['Filename'].str.contains('ctr')),:]))
        print("FP DN CTR", len(
            ftl.loc[(ftl['Triage_Score'] == 0) & (ftl['dn_Pred'] > 0.5) & (ftl['Filename'].str.contains('ctr')), :]))

        print("FN TL:", len(fn_tl), " FN DN:", len(fn_dn))
        print("Common FN=", len(pd.merge(fn_dn, fn_tl, how='inner', on=['Filename', 'Slice_Number'])))
        print("FN TL Mild", len(ftl.loc[(ftl['Triage_Score'] == 1) & (ftl['Pred_Score'] <= 0.5) & (ftl['Orig_Score'].str.contains('Mild')),:]))
        print("FN DN Mild", len(
            ftl.loc[(ftl['Triage_Score'] == 1) & (ftl['dn_Pred'] <= 0.5) & (ftl['Orig_Score'].str.contains('Mild')), :]))

    st, p = ttest_ind(fpr_tl, fpr_dn, equal_var=False)
    print("FPR TL", np.mean(fpr_tl), np.std(fpr_tl))
    print("FPR DN: ", np.mean(fpr_dn), np.std(fpr_dn), "P-value ", p)

    st, p = ttest_ind(fpr_tlc, fpr_dnc, equal_var=False)
    print("FPR CTR TL: ", np.mean(fpr_tlc), np.std(fpr_tlc))
    print("FPR CTR DN: ", np.mean(fpr_dnc), np.std(fpr_dnc), "P-val ", p)

    st, p = ttest_ind(fnr_tl, fnr_dn, equal_var=False)
    print("FNR TL: ", np.mean(fnr_tl), np.std(fnr_tl))
    print("FNR DN: ", np.mean(fnr_dn), np.std(fnr_dn), "P-val ", p)

    st, p = ttest_ind(fnr_tlm, fnr_dnm, equal_var=False)
    print("FNR MILD TL: ", np.mean(fnr_tlm), np.std(fnr_tlm))
    print("FNR MILD DN: ", np.mean(fnr_dnm), np.mean(fnr_dnm), "P-val ", p)

