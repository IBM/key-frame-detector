import pandas as pd
import numpy as np
import os
import skimage.transform as skt
import pickle

def gen_pkl_file(df, datapath, INPUT_SHAPE, pkl_filename="all_data_reshaped.pkl"):
    """
    Generates a data dictionary of all the OCT slices.
    The dataframe df contains all the slices of the OCT volumes and their associated
    labels, unique identifiers, and a 10-fold cross validation division into Train,
    Validation and Test sets.
    Once the data has been read in, a data dictionary is created
        'image_data': contains the individual OCT slices resized to (150, 256, 1)
        'label': contains the label '0' for Normal and '1' for Abnormal (mild or severe
        pathologies)
        'uid': is a unique identifier for each slices consisting of the OCT volume
        filename and the slice number. e.g. 1001_ctr_vol_0 is the first slice of the volume
        1001_ctr_vol

    :param df: dataframe
    :param datapath: path to folder containing the OCT volumes
    :param INPUT_SHAPE: size of the slices expected by the deep learning CNN
    :param pkl_filename: path and name of file where the dictionary is stored
    :return: data_dict: dictionary containing the image data, labels and unique identifier
    """
    data_dict = {}
    file_list = df.loc[:, 'Filename'].unique()
    for i in range(0, len(file_list)):
        # print(file_list[i])
        if 100.0 * (i // len(file_list)) > 90 and 100 * (i / len(file_list)) < 90.5:
            print("90%")
        elif 100 * (i // len(file_list)) > 75 and 100 * (i / len(file_list)) < 75.5:
            print("75%")
        elif 100 * (i // len(file_list)) > 50 and 100 * (i / len(file_list)) < 50.5:
            print("50%")
        elif 100 * (i // len(file_list)) > 25 and 100 * (i / len(file_list)) < 25.5:
            print("25%")

        fname = os.path.join(datapath, file_list[i]) + ".npy"
        print(fname)
        im_vol = np.load(fname)
        val_slices = df.loc[(df['Filename'] == file_list[i]), {'Slice_Number'}].values
        uid = df.loc[(df['Filename'] == file_list[i]), 'uid'].values

        im_vol = im_vol[val_slices, :, :]
        l = df.loc[(df['Filename'] == file_list[i]), {'Triage_Score'}].values

        for j in range(0, im_vol.shape[0]):
            # if (i == 0 and j == 0) or len(img_list) == 0:
            imgdat = skt.resize(np.squeeze(im_vol[j, :, :]).astype('float'),
                                    output_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1]), preserve_range=True)
            imgdat = 255*(imgdat)/(np.max(np.max(imgdat)) - np.min(np.min(imgdat)))
            imgdat = np.reshape(imgdat.astype('uint8'), newshape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 1))
            data_dict[uid[j]] = [imgdat, l[j]]

    print("total number of samples: ", len(data_dict))
    """ since individual B-scans are resized, final dataset containing 115 healthy and 269 AMD eyes 
    generates a pkl file 9.4GB in size"""
    pickle.dump(data_dict, open(pkl_filename, 'wb'))
    return data_dict