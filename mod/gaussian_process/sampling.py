# -*- coding: utf-8 -*-
"""
Created on 2020/1/30 18:03

@Project -> File: gaussian-process-regression -> generate_gp_samples.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 生成高斯过程样本
"""

import seaborn as sns
import numpy as np
import sys

sys.path.append('../')

from mod.gaussian_process.kernels import kernel_func


def gpr_sampling(t_list, kernel_name, kernel_params, show_matrix_C = False):
	"""
	高斯过程采样
	这部分过程和原理可以参考以下介绍高斯过程采样算法的材料:
		https://www.jgoertler.com/visual-exploration-gaussian-processes/#GaussianProcesses
		https://blog.csdn.net/shenxiaolu1984/article/details/50386518
	:param t_list: array like, 时间值序列
	:param kernel_name: str, 高斯过程核函数名称
	:param kernel_params: dict, 高斯过程核函数参数
	:param show_matrix_C: bool, 显示相关矩阵C
	
	Example:
	------------------------------------------------------------
	t_list = np.arange(0, 10, 0.1)
	kernel_name = 'RBF'
	kernel_params = {'l': 2.0, 'p': 3}
	x_list = gpr_sampling(t_list, kernel_name = 'RBF', kernel_params = kernel_params, show_matrix_C = True)
	------------------------------------------------------------
	"""
	t_list = np.array(t_list).flatten()
	
	# 生成采样点序列x.
	N = len(t_list)
	
	# 通过高斯核函数计算采样点之间的相关函数矩阵C.
	C = np.zeros([N, N])
	for i in range(N):
		for j in range(i, N):
			t_i, t_j = t_list[i], t_list[j]
			C[i, j] = kernel_func(t_i, t_j, kernel_name, kernel_params)
	C = C + np.tril(C.T, -1)  # **高斯核函数具有对称性质, 所以此处为矩阵与转置矩阵下三角之和
	
	if show_matrix_C:
		sns.heatmap(C)
	
	# 对C进行SVD分解.
	U, sigmas, V = np.linalg.svd(C)
	S = np.diag(sigmas)  # 向量转为对角矩阵
	
	# 生成N个独立同分布高斯随机变量.
	y_list = np.random.normal(loc = 0.0, scale = 1.0, size = (N,))
	x_list = np.dot(np.dot(U, np.sqrt(S)), y_list.reshape(-1, 1))
	return x_list
	
	
	
	

