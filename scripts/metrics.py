import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



def angle(pred, true):
    return tf.math.acos(
        tf.clip_by_value(
            tf.math.divide_no_nan(tf.reduce_sum(pred * true, axis = 1),
            tf.math.reduce_euclidean_norm(pred, axis = 1) * tf.math.reduce_euclidean_norm(true,  axis = 1)),
            -1., 1.)
        )

def angle_u(y_true, y_reco):
    # Angle metric
    angle_resi = 180 / np.pi * angle(y_reco[:, :3], y_true[:, :3])

    u_angle         = tfp.stats.percentile(angle_resi, [68])

    return float(u_angle.numpy())