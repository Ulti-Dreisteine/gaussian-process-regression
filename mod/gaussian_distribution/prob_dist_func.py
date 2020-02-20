# -*- coding: utf-8 -*-
"""
Created on 2020/2/20 21:07

@Project -> File: gaussian-process-regression -> prob_dist_func.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 概率密度函数
"""

import numpy as np


def one_dim_gaussian(x, mu, sigma):
	"""
	一维高斯分布概率密度函数
	:param x: float or array like, x值或向量
	:param mu: float, 均值
	:param sigma: float, 标准差
	"""
	expo = np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))
	f = 1 / (sigma * np.sqrt(2 * np.pi)) * expo
	return f


def multi_dim_gaussian(x, mu, Sigma):
	"""
	多维高斯分布概率密度函数
	:param x: array like, x向量
	:param mu: array like, 均值向量
	:param Sigma: np.array, 协方差矩阵
	:return:
	"""
	x = np.array(x).reshape(-1, 1)
	mu = np.array(mu).reshape(-1, 1)
	dim_x = x.shape[0]
	
	try:
		assert mu.shape[0] == dim_x
		assert Sigma.shape[0] == Sigma.shape[1] == dim_x
	except:
		raise ValueError('The shape of mu or Sigma does not correspond to the dimension of x')
	
	expo = np.exp(-0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(Sigma)), x - mu))
	f = 1 / (np.power(2 * np.pi, dim_x / 2) * np.power(np.linalg.det(Sigma), 0.5)) * expo
	return f[0]


if __name__ == '__main__':
	import logging
	
	logging.basicConfig(level = logging.INFO)
	
	import matplotlib.pyplot as plt
	
	# %% 一维高斯分布.
	mu = 0.0
	sigma = 1.0
	x = np.arange(-10.0, 10.0 + 0.1, 0.1).reshape(-1, 1)
	y = one_dim_gaussian(x, mu, sigma)
	plt.figure('One Dimensional PDF')
	plt.plot(x, y)
	
	# %% 二维高斯分布.
	from mod.gaussian_process.sampling import cal_covariance_matrix
	mu = [3.0, 1.0]
	idx = [0.0, 1.0]
	kernel_name = 'RBF'
	kernel_params = {'sigma': 2.0, 'l': 1.0}
	Sigma = cal_covariance_matrix(idx, kernel_name = kernel_name, kernel_params = kernel_params)
	
	x = np.arange(-10.0, 10.0 + 0.2, 0.2).reshape(-1, 1)
	y = np.arange(-10.0, 10.0 + 0.2, 0.2).reshape(-1, 1)
	mesh_x, mesh_y = np.meshgrid(x, y)
	coords = np.dstack((mesh_x, mesh_y))
	pdf = np.apply_along_axis(lambda x: multi_dim_gaussian(x, mu, Sigma), 2, coords)
	pdf = pdf.reshape(len(x), len(y))  # TODO: 确认pdf中的维度与x, y维度对应
	plt.figure('Two Dimensional PDF')
	plt.contourf(mesh_x, mesh_y, pdf, cmap = 'Blues')
	