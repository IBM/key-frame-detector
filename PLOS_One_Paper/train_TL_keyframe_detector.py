import pandas as pd
import numpy as np
import argparse
import os.path
import keras.callbacks as kc
import random as rnd
import pickle
import keras.utils as ktil

import custom_network as cnet
import utils
from constants import *

def read_meta_data_file(filename, verbose=1):
    """
        Read metadata file, encode Triage_Score into numerals
        Parameters:
            filename: file to be read
            verbose (1 or 0): 1 - do not print anything, 0 - print length of dataframe and head

        returns:
            cleaned up dataframe
    """
    if filename.endswith('.csv'):
        raw_data_df = pd.read_csv(filename)
    elif filename.endswith('xlsx'):
        raw_data_df = pd.read_excel(filename)
    # print(len(raw_data_df), raw_data_df.head())
    data_df = raw_data_df.dropna(subset=['Triage_Score'])

    if verbose == 0:
        print(len(data_df), data_df.head())

    return data_df

def generate_random_training_samples(train_data, batch_size):
    """
        Generate random set of training samples

        Parameters:
             val_data_dict: dictionary containing 'image_data', 'labels' and
                'image_name' (unique identifier)
             batch_size: number of samples in each batch

        Returns:
            list of samples and labels
    """
    while True:
        n_keys = rnd.sample(train_data.keys(), batch_size)
        im_shape = train_data[n_keys[0]][0].shape
        train_im = np.zeros(shape=(batch_size, im_shape[0], im_shape[1],1))
        t_label = np.zeros(shape=(batch_size,1))
        for m in range(len(n_keys)):
            train_im[m,:,:,:] = train_data[n_keys[m]][0].astype('float')/255.0
            t_label[m] = train_data[n_keys[m]][1]
        train_label = ktil.to_categorical(t_label, NUM_CLASSES)
        yield train_im, train_label


def generate_random_validation_samples(val_data, batch_size):
    """
        Generate random set of validation samples

        Parameters:
             val_data_dict: dictionary containing 'image_data', 'labels' and
                'image_name' (unique identifier)
             batch_size: number of samples in each batch

        Returns:
            list of samples and labels
    """
    while True:
        n_keys = rnd.sample(val_data.keys(), batch_size)
        im_shape = val_data[n_keys[0]][0].shape
        val_im = np.zeros(shape=(batch_size, im_shape[0], im_shape[1], 1))
        v_label = np.zeros(shape=(batch_size, 1))
        for m in range(batch_size):
            val_im[m, :, :, :] = val_data[n_keys[m]][0].astype('float')/255.0
            v_label[m] = val_data[n_keys[m]][1]
        val_label = ktil.to_categorical(v_label, NUM_CLASSES)
        yield val_im, val_label



if __name__ == "__main__":
    num_folds = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_data_filename', default="0", help="Enter csv with label info")
    parser.add_argument('--data_path', default='0', help='Enter path to folder containing images')
    parser.add_argument("--data_pkl_file", default='0', help='Enter path to pkl containing data')

    args = parser.parse_args()

    if args.data_path == '0' or not os.path.isdir(args.data_path):
        print("Cannot proceed without valid data path")
        print("Usage: train_denovo_keyframe_detector.py --meta_data_filename {filename} --data_path {path} "
              "--data_pkl_file {filename}")
        exit(1)

    if args.meta_data_filename == '0' or not os.path.exists(args.meta_data_filename):
        print("Cannot proceed without meta data file")
        print("Usage: train_denovo_keyframe_detector.py --meta_data_filename {filename} --data_path {path}"
              "--data_pkl_file {filename}")
        exit(1)

    data_df = read_meta_data_file(args.meta_data_filename, verbose=1)
    if args.data_pkl_file == '0' or not os.path.exists(args.data_pkl_file):
        print("beginning generation of pkl. If data has already been curated into pkl file, pls provide path")
        img_labeled_data = utils.gen_pkl_file(data_df, args.data_path, INPUT_SHAPE, pkl_filename=args.data_pkl_file)
    else:
        print("Reading pkl file")
        img_labeled_data = pickle.load(open(args.data_pkl_file, "rb"))

    """ Extract samples from training, test and validation sets"""
    for k in range(num_folds):
        split_name = 'split_label' + str(k)
        train_uid = data_df.loc[data_df[split_name] == 'Training', 'uid'].values
        val_uid = data_df.loc[data_df[split_name] == 'Validation', 'uid'].values

        train_data = {}
        val_data = {}

        for i in train_uid:
            train_data[i] = img_labeled_data[i]

        for i in val_uid:
            val_data[i] = img_labeled_data[i]

        print("training, val, split = (", len(train_data), len(val_data), ")")

        """ Create model with appropriate call backs"""
        weights_path = "TL_weights" + str(k) + ".h5"
        model = cnet.create_TLnetwork_VGG16(INPUT_SHAPE, NUM_CLASSES, weights_path=weights_path)
        checkpoint = kc.ModelCheckpoint(filepath=weights_path, monitor="val_loss",
                                        save_best_only=True, mode="min", save_weights_only=True)

        reduceLR = kc.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, verbose=0, mode='min',
                                    cooldown=0, min_lr=1e-15)
        stopcriteria = kc.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=50, verbose=0, mode='min')
        """" training """
        steps_per_epoch_train = len(train_uid)//BATCH_SIZE + 1
        steps_per_epoch_val = len(val_uid) // BATCH_SIZE + 1
        hist_ls = model.fit_generator(generate_random_training_samples(train_data, BATCH_SIZE),
                            steps_per_epoch=steps_per_epoch_train,
                            epochs=200,
                            validation_data=generate_random_validation_samples(val_data, BATCH_SIZE), verbose=2,
                            validation_steps=steps_per_epoch_val, callbacks=[checkpoint, stopcriteria], shuffle=True)

        hist = pd.DataFrame(data=hist_ls.history)
        logfile_name = 'train_TL_log' + str(k) + ".csv"
        hist.to_csv(logfile_name, index=True)
        "load best checkpointed weights"
        model.load_weights(weights_path)

        "test on unseen data, i.e. test data"
        test_data = data_df.loc[data_df['Set_Label'].str.contains('Testing'),:]
        test_data['Prediction'] = np.zeros(len(test_data))
        for i, row in test_data.iterrows():
            uid = row['uid']
            p = model.predict(np.reshape(img_labeled_data[uid][0], newshape=(1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1)))
            # print(uid)
            test_data.loc[i, 'Prediction'] = 2.0

        print('writing file')
        outfilename = 'TL_testSet' + str(k) + '.csv'
        test_data.to_csv(outfilename, index=False)