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

    return u_angle.numpy()


def pull_z(y_true, y_reco):
    angle_resi = angle(y_reco[:, :3], y_true[:, :3])
    k    = y_reco[:, 3]
    sig        = tf.math.divide_no_nan(1., tf.math.sqrt(k)) 

    pull       = angle_resi / sig

    return tfp.stats.percentile(pull, [68])



def sigma_median(y_true, y_reco):
    # Return the median uncertainity from the predictions
    k    = y_reco[:, 3]
    sig  = tf.math.divide_no_nan(1., tf.math.sqrt(k))

    return 180 / np.pi * tfp.stats.percentile(sig, [50])


def zenith_u(y_true, y_reco):
    # Returns the angle metric for the zenith angle
    zenith_true  = tf.math.acos(tf.clip_by_value(
        y_true[:, 2],
        -1, 1))
    zenith_pred  = tf.math.acos(tf.clip_by_value(
        y_reco[:, 2],
        -1, 1))

    diffs        = tf.abs(zenith_pred - zenith_true)

    u_zenith     = tfp.stats.percentile(diffs, [68])
    
    return 180 / np.pi * u_zenith


def azimuthal_u(y_true, y_reco):
    # Returns the angle metric for the zenith angle
    azimuthal_true  = tf.math.atan(tf.math.divide_no_nan(y_true[:, 1],  y_true[:, 0]))
    azimuthal_pred  = tf.math.atan(tf.math.divide_no_nan(y_reco[:, 1],  y_reco[:, 0]))

    diffs        = tf.abs(azimuthal_true - azimuthal_pred)

    diffs        = tf.math.minimum(diffs, 2 * np.pi - diffs)

    u_azi        = tfp.stats.percentile(diffs, [68])
    
    return 180 / np.pi * u_azi
