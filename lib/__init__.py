# -*- coding: utf-8 -*-
"""
Created on 2020/1/30 17:02

@Project -> File: gaussian-process-regression -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

import numpy as np
import sys

sys.path.append('../')

from mod.config.config_loader import config
from mod.gaussian_process.sampling import gpr_sampling

proj_dir = config.proj_dir
proj_cmap = config.proj_cmap


def gen_gaussian_process_samples(t_list, mu, C, samples_n):
	"""
	生成高斯过程样本
	:param t_list: array like, 时刻list
	:param mu: np.array, 均值向量
	:param C: np.array, 协方差举证
	:param sample_n: int, 采样数
	"""
	t_list = np.array(t_list).reshape(-1, 1)
	mu = np.array(mu).reshape(-1, 1)
	
	dims = [t_list.shape[0], mu.shape[0], C.shape[0], C.shape[1]]
	if len(set(dims)) != 1:
		raise ValueError('t_list, mu和C的维度不一致.')
	
	samples = None
	for i in range(samples_n):
		x = gpr_sampling(t_list, mu, C)
		if i == 0:
			samples = x
		else:
			samples = np.hstack((samples, x))
	return samples



