from keras import backend as K
import tensorflow as tf
import numpy as np

# https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
"""
Define our custom loss function.
"""

def BCE_plus_dice(ce_weight, dice_weight):
    def BCE_plus_dice_fixed(y_true, y_pred):
        return ce_weight * K.binary_crossentropy(y_true, y_pred) + dice_weight * dice_coef_loss(y_true, y_pred)
    return BCE_plus_dice_fixed

def BCE_plus_IOU(ce_weight, iou_weight):
    def BCE_plus_dice_fixed(y_true, y_pred):
        return ce_weight * K.binary_crossentropy(y_true, y_pred) + iou_weight * IoU(y_true, y_pred)
    return BCE_plus_dice_fixed

def BCE_plus_tversky(ce_weight, tversky_weight):
    def BCE_plus_tversky_fixed(y_true, y_pred):
        return ce_weight * K.binary_crossentropy(y_true, y_pred) + tversky_weight * tversky_loss(y_true, y_pred)
    return BCE_plus_tversky_fixed

def BCE_plus_tverskyfocal(ce_weight, tverskyfocal_weight):
    def BCE_plus_tverskyfocal_fixed(y_true, y_pred):
        return ce_weight * K.binary_crossentropy(y_true, y_pred) + tverskyfocal_weight * focal_tversky_loss(y_true, y_pred)
    return BCE_plus_tverskyfocal_fixed

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

# dice loss
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.sum( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred, smooth=1)

## intersection over union
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.sum( (intersection + eps) / (union + eps), axis=0)

# tversky loss
def tversky(y_true, y_pred):
    smooth = 1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
def focal_tversky_loss(y_true, y_pred):
    return 1- focal_tversky(y_true, y_pred)

# def tversky_loss_fixed(alpha):
#     def tversky(y_true, y_pred):
#       y_true_pos = K.flatten(y_true)
#       y_pred_pos = K.flatten(y_pred)
#       true_pos = K.sum(y_true_pos * y_pred_pos)
#       false_neg = K.sum(y_true_pos * (1-y_pred_pos))
#       false_pos = K.sum((1-y_true_pos)*y_pred_pos)
#       smooth =1
#       return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
#     def tversky_loss(y_true, y_pred):
#       return 1 - tversky(y_true,y_pred)
#     return tversky_loss

# ------------------------------------lovasz_softmax------------------------------------------

# -------------------------------------------------lovasz_softmax end----------------------------------------------------------------------

'''-------------------------------------------------OLD----------------------------------------------------------------------'''
# alpha = .25
# gamma = 2
# def focal_plus_cross(y_true, y_pred):
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#     return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(tf.clip_by_value(pt_1, 1e-8, 1.0))) - K.mean(
#         (1 - alpha) * K.pow(pt_0, gamma) * K.log(tf.clip_by_value(1. - pt_0, 1e-8, 1.0))) + K.binary_crossentropy(y_true, y_pred)
#
# def focal_loss(y_true, y_pred):
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#     return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(tf.clip_by_value(pt_1, 1e-8, 1.0))) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(tf.clip_by_value(1. - pt_0, 1e-8, 1.0)))

'''-------------------------------------------------------------------------------------------------------------------------------'''
# https://github.com/mkocabas/focal-loss-keras/blob/master/focal_loss.py
# Compatible with tensorflow backend

# def focal_loss(gamma=2., alpha=.25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
#             (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#
#     return focal_loss_fixed

def focal_plus_tversky(Lambda):
    def focal_plus_tversky_fixed(y_true, y_pred):
        focal = focal_loss(gamma= 1, alpha= 0.5)
        tversky = tversky_loss_fixed(alpha= 0.6)
        return Lambda * focal(y_true, y_pred) + tversky(y_true, y_pred)
    return focal_plus_tversky_fixed
