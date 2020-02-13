# -*- coding: utf-8 -*-
"""
Created on 2020/1/30 18:04

@Project -> File: gaussian-process-regression -> kernels.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 高斯过程核函数
"""

import numpy as np


def RBF_kernel(x_a, x_b, sigma = None, l = None):
	"""
	RBF高斯核函数, 又名Exponetiated Quadratic, 公式为:
	k = sigma^2 * exp(-norm(x_a - x_b)^2 / (2 * l^2))
	:param x_a: float, 样本值位置a
	:param x_b: float, 样本值位置b
	:param sigma: float, 方差参数
	:param l: float, 长度参数length
	"""
	if sigma is None:
		sigma = 0.8
	if l is None:
		l = 0.5
		
	n = np.linalg.norm(x_a - x_b)
	k = pow(sigma, 2) * np.exp(- pow(n, 2) / (2 * pow(l, 2)))
	return k


def periodic_kernel(x_a, x_b, sigma = None, l = None, p = None):
	"""
	周期性核函数, 公式为:
	k = sigma^2 * np.exp(-(2 / l^2) * sin(pi / p * |x_a - x_b|)^2)
	:param x_a: float, 样本值位置a
	:param x_b: float, 样本值位置b
	:param sigma: float, 方差参数
	:param l: float > 0.0, 长度参数length
	:param p: flaot > 0.0, 周期参数
	"""
	if sigma is None:
		sigma = 0.8
	if l is None:
		l = 0.5
	if p is None:
		p = 0.5
	
	sin = np.sin(np.pi / p * np.abs(x_a - x_b))
	k = pow(sigma, 2) * np.exp(-2 / pow(l, 2) * pow(sin, 2))
	return k


def linear_kernel(x_a, x_b, sigma = None, sigma_b = None, c = None):
	"""
	线性核函数, 公式为:
	k = sigma_b^2 + sigma^2 * (x_a - c)(x_b - c)
	:param x_a: float, 样本值位置a
	:param x_b: float, 样本值位置b
	:param sigma: float, 方差参数
	:param sigma_b: float, 方差参数b
	:param c: float, offset参数
	"""
	if sigma is None:
		sigma = 0.8
	if sigma_b is None:
		sigma_b = 0.5
	if c is None:
		c = 0.5
	
	k = pow(sigma_b, 2) + pow(sigma, 2) * (x_a - c) * (x_b - c)
	return k



