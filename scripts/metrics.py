import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def azimuthal_u_from_angles(y_true, y_reco):
    diffs = tf.minimum(abs(y_true[:, 0] - y_reco[:, 0]), 2 * np.pi - abs(y_true[:, 0] - y_reco[:, 0]))

    u_azi = 180 / np.pi * tfp.stats.percentile(diffs, [68])

    return u_azi.numpy()



def zenith_u_from_angles(y_true, y_reco):
    diffs = tf.minimum(abs(y_true[:, 1] - y_reco[:, 1]), 2 * np.pi - abs(y_true[:, 1] - y_reco[:, 1]))

    u_zen = 180 / np.pi * tfp.stats.percentile(diffs, [68])

    return u_zen.numpy()



def azimuthal_u_from_cossine(y_true, y_reco):

    aziguess = tf.atan2(y_reco[:,1],y_reco[:,0])
    azi = tf.minimum( tf.abs(y_true[:,0] - aziguess) , tf.abs(tf.abs(y_true[:,0] - aziguess) - 2*np.pi))
    
    u_azi = 180 / np.pi * tfp.stats.percentile(azi, [68])

    return u_azi.numpy()


def zenith_u_from_cossine(y_true, y_reco):
    zeniguess = tf.atan2(y_reco[:,3],y_reco[:,2])
    zeni = tf.minimum( tf.abs(y_true[:,1] - zeniguess) , tf.abs(tf.abs(y_true[:,1] - zeniguess) - 2*np.pi))

    u_zenth = 180 / np.pi *  tfp.stats.percentile(zeni, [68])
    
    return u_zenth.numpy()


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


def mean_sigma_zenith(y_true, y_reco):
    polar_k   = y_reco[:, 4]

    return tf.reduce_mean(1 / tf.sqrt(polar_k)) * 180 / np.pi

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


def mean_zenith_std(y_true, y_reco):
    zenith_k   = y_reco[:, 4]

    std       = tf.math.sqrt(- 2 * tf.math.log(tf.math.divide_no_nan(tf.math.special.bessel_i1(zenith_k),
                                                       tf.math.special.bessel_i0(zenith_k))))

    return tf.reduce_mean(std) / np.pi * 180


def mean_azimuth_std(y_true, y_reco):
    polar_k   = y_reco[:, 3]

    std       = tf.math.sqrt(- 2 * tf.math.log(tf.math.divide_no_nan(tf.math.special.bessel_i1(polar_k),
                                                       tf.math.special.bessel_i0(polar_k))))

    return tf.reduce_mean(std) / np.pi * 180


from tensorflow.keras.metrics import BinaryAccuracy
from sklearn.metrics import roc_auc_score

def AUC(y_true, y_reco):
    auc = roc_auc_score(y_true.numpy(), tf.y_reco.numpy())
    return auc

BA = BinaryAccuracy()
def binary_accuracy(y_true, y_reco):
    return BA(y_true, y_reco)
