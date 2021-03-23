
import os
import numpy as np
import skimage.transform as skt
import argparse

from keras.models import Model
from keras.layers import (Input, Conv2D, Concatenate, MaxPooling2D, Dense, GlobalAveragePooling2D)
from keras.optimizers import SGD, Adam
import keras.backend as K

INPUT_SHAPE = (150, 256, 1)
H,W = (300, 512)

BATCH_SIZE = 64
NUM_CLASSES = 2



def class_balanced_cross_entropy_loss(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)

    y_true_f_n = 1.0 - y_true_f
    y_pred_f_n = 1.0 - y_pred_f

    Y_plus = K.sum(y_true_f, axis=1, keepdims=True)
    Y_minu = K.sum(y_true_f_n, axis=1, keepdims=True)
    Y = Y_minu + Y_plus

    p_loss = (Y_minu / Y) * K.sum(K.log(y_pred_f + K.epsilon()) * y_true_f, axis=-1, keepdims=True) / Y
    n_loss = (Y_plus / Y) * K.sum(K.log(y_pred_f_n + K.epsilon()) * y_true_f_n, axis=-1, keepdims=True) / Y

    return (- p_loss - n_loss) * 100.

def create_denovo_network(INPUT_SHAPE, NUM_CLASSES, weights_path="temp.h5"):

    in_im = Input(shape=INPUT_SHAPE)
    l1 = Conv2D(32, (3, 3), dilation_rate=(2, 2), padding='same', activation='relu', use_bias=True, name='L1')(in_im)
    c1 = Concatenate()([in_im, l1])
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    l2 = Conv2D(64, (3, 3), dilation_rate=(2, 2), padding='same', activation='relu', use_bias=True, name='L2')(p1)
    c2 = Concatenate()([p1, l2])
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    l3 = Conv2D(128, (3, 3), dilation_rate=(2, 2), padding='same', activation='relu', use_bias=True, name='L3')(p2)
    c3 = Concatenate()([p2, l3])
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    l4 = Conv2D(128, (3, 3), dilation_rate=(2, 2), padding='same', activation='relu', use_bias=True, name='L4')(p3)

    f = GlobalAveragePooling2D(name="CAM")(l4)
    out = Dense(NUM_CLASSES, activation='softmax', name='out1')(f)

    model = Model(inputs=[in_im], outputs=[out])

    adam = Adam(lr=1e-06, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=1e-08, momentum=0.5, decay=1e-11, nesterov=False)
    model.compile(loss=[class_balanced_cross_entropy_loss],
                      optimizer=sgd,
                      metrics=['accuracy'])
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
    # print(model.summary())

    return model


def create_TLnetwork_VGG16(INPUT_SHAPE, NUM_CLASSES, weights_path):
    from keras.applications.vgg16 import VGG16
    from keras.models import Model
    from keras.layers import Flatten
    base_model = VGG16(input_shape=INPUT_SHAPE,weights='imagenet', include_top=False,pooling='avg')
    base_model.layers.pop()

    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-1].output
    # ayer['block5_pool'].output
    # cv1 = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)
    f = GlobalAveragePooling2D(name="CAM")(x)
    fc1 = Dense(1024, activation='relu')(f)
    fc2 = Dense(512, activation='relu')(fc1)
    predictions = Dense(NUM_CLASSES, activation='softmax', name='out')(fc2)
    model = Model(inputs=[base_model.input], outputs=[predictions])

    adam = Adam(lr=1e-06, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0)
    sgd = SGD(lr=1e-015, momentum=0.5, decay=1e-21, nesterov=False)
    model.compile(loss=[class_balanced_cross_entropy_loss],
                  optimizer=adam,
                  metrics=['accuracy'])
    # plot_model(model, to_file='TL.png', show_shapes=True)
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
    print(model.summary())
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="DN", help='Enter model type. '
                                                           'DN: denovo model, TL: transfer learning model')
    parser.add_argument('--mode_weights', default='0', help='Enter path to model weights')
    parser.add_argument('--input_file', default='0', help="Enter path and filename of npy volume to be processed.")

    args = parser.parse_args()
    model_type = args.model_type
    weightpath = args.model_weights
    if weightpath == '0' or not os.path.isfile(weightpath):
        print("Cannot proceed without weights file")
        exit(1)

    input_vol = args.input_file
    if input_vol == '0' or not os.path.isfile(input_vol):
        print('Cannot proceed without input file')
        exit(1)
    elif not input_vol.endswith('.npy'):
        print('Please provide OCT volume as numpy array in format Bscans x Depth x Width')
        exit(1)


    if model_type.find("DN") > -1:
        """ de novo network"""
        INPUT_SHAPE = INPUT_SHAPE
        model = create_denovo_network(INPUT_SHAPE, NUM_CLASSES, weights_path=weightpath)
        replicate_flag = 0
    else:
        """ TL """
        INPUT_SHAPE = INPUT_SHAPE
        model = create_TLnetwork_VGG16(INPUT_SHAPE, NUM_CLASSES, weights_path=weightpath)
        replicate_flag=1

    im_vol = np.load(input_vol)
    img_data = np.zeros(shape=(im_vol.shape[0], INPUT_SHAPE[0], INPUT_SHAPE[1], 1))
    for j in range(im_vol.shape[0]):
        tmp = skt.resize(np.squeeze(im_vol[j, :, :]).astype('float'),
                         output_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1]), preserve_range=True)
        tmp = (tmp) / (np.max(np.max(tmp)) - np.min(np.min(tmp)))
        tmp = np.reshape(tmp, newshape=(INPUT_SHAPE))
        img_data[j, :,:,:] = tmp


    for j in range(0, im_vol.shape[0], BATCH_SIZE):
        """ set last index of images from list that are to be processed """
        fin_ind = j + BATCH_SIZE
        if fin_ind >= im_vol.shape[0]:
            fin_ind = im_vol.shape[0] - 1

        test_pred = model.predict(img_data[j:fin_ind], batch_size=BATCH_SIZE, verbose=2)
        t_pred = test_pred[:, 1]
        print(t_pred)




