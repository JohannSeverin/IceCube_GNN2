import tensorflow as tf
from tensorflow.keras.backend import epsilon

import tensorflow_probability as tfp

from numpy import pi

eps = 1e-5




def VonMises3D(y_true, y_reco):
    k = y_reco[:, 3]
    vects = y_reco[:, :3]
    log_likelihood =  - k / 2 * tf.squeeze(tf.expand_dims(vects - y_true, axis = 1) @ tf.expand_dims(vects - y_true, axis = -1))\
                      + tf.math.log(k) - tf.math.log(1 - tf.exp(- 2 * k))
                     

    return tf.reduce_mean(- log_likelihood)