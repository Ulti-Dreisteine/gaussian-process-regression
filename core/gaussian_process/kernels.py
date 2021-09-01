# -*- coding: utf-8 -*-
"""
Created on 2020/1/30 18:04

@Project -> File: gaussian-process-regression -> kernels.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 高斯过程核函数
"""

import numpy as np


def _rbf_kernel(x_a, x_b, sigma: float = 0.8, l: float = 0.5):
    """
    RBF高斯核函数, 又名Exponetiated Quadratic, 公式为:
    k = sigma^2 * exp(-norm(x_a - x_b)^2 / (2 * l^2))
    :param x_a: float, 样本值位置a
    :param x_b: float, 样本值位置b
    :param sigma: float, 方差参数
    :param l: float, 长度参数length
    """
    n = np.linalg.norm(x_a - x_b)
    k = pow(sigma, 2) * np.exp(- pow(n, 2) / (2 * pow(l, 2)))
    return k


def _periodic_kernel(x_a, x_b, sigma: float = 0.8, l: float = 0.5, p: float = 0.5):
    """
    周期性核函数, 公式为:
    k = sigma^2 * np.exp(-(2 / l^2) * sin(pi / p * |x_a - x_b|)^2)
    :param x_a: float, 样本值位置a
    :param x_b: float, 样本值位置b
    :param sigma: float, 方差参数
    :param l: float > 0.0, 长度参数length
    :param p: flaot > 0.0, 周期参数
    """
    assert (l > 0.0) & (p > 0.0)
    sin = np.sin(np.pi / p * np.abs(x_a - x_b))
    k = pow(sigma, 2) * np.exp(-2 / pow(l, 2) * pow(sin, 2))
    return k


def _linear_kernel(x_a, x_b, sigma: float = 0.8, sigma_b: float = 0.5, c: float = 0.5):
    """
    线性核函数, 公式为:
    k = sigma_b^2 + sigma^2 * (x_a - c)(x_b - c)
    :param x_a: float, 样本值位置a
    :param x_b: float, 样本值位置b
    :param sigma: float, 方差参数
    :param sigma_b: float, 方差参数b
    :param c: float, offset参数
    """
    k = pow(sigma_b, 2) + pow(sigma, 2) * (x_a - c) * (x_b - c)
    return k


def calKernelFunc(x_a, x_b, kernel_name, **kwargs):
    """核函数计算"""

    if kernel_name == 'RBF':
        k = _rbf_kernel(x_a, x_b, **kwargs)
        return k
    elif kernel_name == 'linear':
        k = _linear_kernel(x_a, x_b, **kwargs)
        return k
    elif kernel_name == 'periodic':
        k = _periodic_kernel(x_a, x_b, **kwargs)
        return k
    else:
        raise ValueError('Unknown kernel func name "{}".'.format(kernel_name))
