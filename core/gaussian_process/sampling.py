# -*- coding: utf-8 -*-
"""
Created on 2021/09/01 10:37:22

@File -> sampling.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 3))
sys.path.append(BASE_DIR)

from core.gaussian_process.kernels import calKernelFunc

__doc__ = """
    参考文献: 
        https://www.jgoertler.com/visual-exploration-gaussian-processes/#GaussianProcesses
		https://blog.csdn.net/shenxiaolu1984/article/details/50386518
"""

def calCovMatrix(t_series: np.ndarray, kernel_name: str, **kernel_params):
	"""计算协方差矩阵C"""
	# 生成采样点序列x.
	N = len(t_series)
	
	# 通过高斯核函数计算采样点之间的相关函数矩阵C.
	C = np.zeros([N, N])
	for i in range(N):
		for j in range(i, N):
			t_i, t_j = t_series[i], t_series[j]
			C[i, j] = calKernelFunc(t_i, t_j, kernel_name, **kernel_params)
	C = C + np.tril(C.T, -1)  # **高斯核函数具有对称性质, 所以此处为矩阵与转置矩阵下三角之和
	return C


def _execGPRSampling(t_series, mu, C) -> np.ndarray:
	"""
	根据给定的时间, 各时间步之间计算所得均值和协方差矩阵进行'单次'高斯过程采样
	这部分过程和原理可以参考以下介绍高斯过程采样算法的材料:
	:param t_series: 时间值序列
	:param mu: 均值向量
	:param C: 协方差矩阵
	
	Example:
	------
	t_series = np.arange(0, 10, 0.1)
	x_series = gpr_sampling(t_series, mu, C)
	"""
	t_series = t_series.flatten()
	
	# 对C进行SVD分解.
	U, sigmas, _ = np.linalg.svd(C)
	S = np.diag(sigmas)  # 向量转为对角矩阵
	
	# 生成N个独立同分布高斯随机变量.
	y_series = np.random.normal(loc = 0.0, scale = 1.0, size = (len(t_series),))
	x_series = np.dot(np.dot(U, np.sqrt(S)), y_series.reshape(-1, 1))
	
	# 加上均值, 得到样本向量.
	x_series += mu
		
	return x_series


def genGaussianProcessSamples(t_series: np.ndarray, mu: np.ndarray, C: np.ndarray, samples_n):
	"""
	生成高斯过程样本集
	:param t_series: 时刻series, shape = (Nt, 1)
	:param mu: np.array, 均值向量, shape = (Nt, 1)
	:param C: np.array, 协方差矩阵, shape = (Nt, N)
	:param sample_n: int, 采样数
	:return samples, shape = (Nt, Ns)

	Example:
	------
	t_series = np.array([0.0, 1.0, 3.0, 4.0, 7.0])
    mu = np.zeros_like(t_series)
    C = np.random.random((t_series.shape[0], t_series.shape[0]))
    samples = genGaussianProcessSamples(t_series, mu, C, samples_n = 10)
	"""
	t_series = t_series.reshape(-1, 1)
	mu = mu.reshape(-1, 1)
	
    # 维数检查.
	dims = [t_series.shape[0], mu.shape[0], C.shape[0], C.shape[1]]
	assert len(set(dims)) == 1
	
	samples = None
	for i in range(samples_n):
		x = _execGPRSampling(t_series, mu, C)
		if i == 0:
			samples = x
		else:
			samples = np.hstack((samples, x))
	return samples


if __name__ == '__main__':
    ...


    