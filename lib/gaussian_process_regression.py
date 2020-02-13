# -*- coding: utf-8 -*-
"""
Created on 2020/2/13 14:08

@Project -> File: gaussian-process-regression -> gaussian_process_regression.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 高斯过程回归模型
"""

import logging

logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../')

from lib import proj_cmap
from mod.gaussian_process.sampling import gpr_sampling


def gen_prior_samples(N):
	"""生成先验样本"""
	t_list = np.arange(0, 10, 0.1)
	kernel_name = 'RBF'
	kernel_params = {'l': 2.0, 'p': 3}
	
	xs = None
	for i in range(N):
		x = gpr_sampling(t_list, kernel_name = kernel_name, kernel_params = kernel_params)
		
		if i == 0:
			xs = x
		else:
			xs = np.hstack((xs, x))
	return xs


if __name__ == '__main__':
	xs = gen_prior_samples(N = 50)
	plt.plot(xs, c = proj_cmap['grey'], linewidth = 0.3)
	plt.plot(np.mean(xs, axis = 1), c = proj_cmap['blue'], linewidth = 2.0)
	
	


