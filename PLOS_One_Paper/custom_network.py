from keras.models import Model
from keras.layers import (Input, Conv2D, Concatenate, MaxPooling2D, Dense, GlobalAveragePooling2D)
from keras.optimizers import SGD, Adam
import keras.backend as K
import os.path
from constants import *

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