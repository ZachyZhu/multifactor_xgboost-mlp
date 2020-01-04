from utils.basic_layer import *
import numpy as np
import tensorflow as tf
import time

def mlp(is_training,hidden_layer):
    input_x=tf.keras.Input(shape=(121))
    fc_layer_0=fc_layer(input_x,input_x.shape[-1],hidden_layer[0],'relu')
    bn_layer_0 = batch_norm_layer(input_x=fc_layer_0.output(),
                                  is_training=is_training)
    fc_layer_1=fc_layer(bn_layer_0.output(),bn_layer_0.output().shape[-1],hidden_layer[1],'relu')
    dropout_0=dropout(fc_layer_1.output(),0.3)
    fc_layer_2=fc_layer(dropout_0.output(),dropout_0.output().shape[-1],10,'softmax')
    cell_out = fc_layer_2.output()

    # model
    model = tf.keras.Model(inputs=input_x, outputs=cell_out)
    return model
