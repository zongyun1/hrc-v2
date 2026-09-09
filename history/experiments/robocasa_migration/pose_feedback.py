"""Rigid held-object pose retargeting, with xyzw quaternion convention."""
import numpy as np


def multiply(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3], b[:3]),
                 a[3]*b[3]-np.dot(a[:3], b[:3])]


def inverse(q):
    q = np.asarray(q)
    return q*np.array([-1., -1., -1., 1.])/np.dot(q, q)


def rotate(q, vector):
    return multiply(multiply(q, np.r_[vector, 0.]), inverse(q))[:3]


def blend_quat(a, b, fraction):
    a, b = np.asarray(a), np.asarray(b)
    if np.dot(a, b) < 0:
        b = -b
    q = (1-fraction)*a+fraction*b
    return q/np.linalg.norm(q)


def held_object_target(eef_pos, eef_quat, object_pos, object_quat, target_pos, target_quat):
    """Preserve the measured object-to-gripper transform when retargeting."""
    delta = multiply(target_quat, inverse(object_quat))
    return (np.asarray(target_pos)-rotate(delta, np.asarray(object_pos)-eef_pos),
            multiply(delta, eef_quat))
