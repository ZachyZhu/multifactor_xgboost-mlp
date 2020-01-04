import numpy as np
import tensorflow as tf

class fc_layer():
    def __init__(self,input_x, in_size, out_size,  activation_function=None):
        """
        :param input_x: The input of the FC layer. It should be a flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param keep_prob: The probability of dropout. Default set by 1.0 (no drop-out applied)
        :param activation_function: The activation function for the output. Default set to None.
        :param index: The index of the layer. It is used for naming only.
        """
        layer = tf.keras.layers.Dense(
                                      units=out_size,
                                      activation=activation_function
        )
        self.cell_out = layer(input_x)

    def output(self):
        return self.cell_out

class batch_norm_layer():
    def __init__(self, input_x, is_training):
        """
        :param input_x: The input that needed for normalization.
        :param is_training: To control the training or inference phase
        """
        layer = tf.keras.layers.BatchNormalization()
        self.cell_out = layer(inputs=input_x,
                              training=is_training)

    def output(self):
        return self.cell_out


class dropout():
     def __init__(self,input_x, keep_prob):
             outputs=tf.nn.dropout(input_x, keep_prob)
             self.cell_out=outputs

     def output(self):
         return self.cell_out