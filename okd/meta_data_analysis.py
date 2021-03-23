import pandas as pd
import numpy as np
import random as rnd
"""
    This script reorganises a few details within the Duke_keyframe_table.csv file. 
    The Duke_keyfame_table.csv file contains the labels Normal, Mild, Severe or NA
    for each slice within the 115 control datasets and 269 AMD datasets from the 
    Duke dataset  http://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm 
    
    The initial labels contained in Duke_keyframe_table.csv was done manually

"""

if __name__ == "__main__":
    df = pd.read_csv('Duke_keyframe_table.csv')
    """ 1. set labels of 0 - Normal, 1 - Abnormal and discard all NA rows

        2. generate a unique identifier for each row filename_slicenumber
    """
    df['uid'] = ""
    for i, row in df.iterrows():
        if row['Triage_Score'] == 'Normal':
            df.loc[i, 'Triage_Score'] = 0
        elif row['Triage_Score'] == 'Mild' or row['Triage_Score'] == 'Severe':
            df.loc[i, 'Triage_Score'] = 1
        else:
            df.loc[i, 'Triage_Score'] = np.nan

        df.loc[i, 'uid'] = row['Filename'] + "_" + str(row['Slice_Number'])

    """ Drop rows without a label, i.e. Ungradable Bscans"""
    df_updated = df.dropna(subset=['Triage_Score'])


    """ To create the 10 individual splits, we divide the subjects into 3 groups - training, validation and 
    test 10 times"""
    subjects = df_updated.loc[:, 'Filename'].unique()

    num_folds = 10
    for i in range(num_folds):
        split_name = 'split_label' + str(i)
        df_updated[split_name] = ""

        train_subjs = rnd.sample(list(subjects), int(0.75 * len(subjects)))

        val_test_subjs = [j for j in subjects if j not in train_subjs]
        val_subjs = rnd.sample(list(val_test_subjs), int(0.40 * len(val_test_subjs)))
        test_subjs = [j for j in val_test_subjs if j not in val_subjs]

        df_updated.loc[df_updated['Filename'].isin(train_subjs), split_name] = 'Training'
        df_updated.loc[df_updated['Filename'].isin(val_subjs), split_name] = 'Validation'
        df_updated.loc[df_updated['Filename'].isin(test_subjs), split_name] = 'Testing'

    df_updated.to_excel('Duke_keyframe_crossval_sets.xlsx', index=False)
