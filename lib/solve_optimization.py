# -*- coding: utf-8 -*-
"""
Created on 2020/2/14 14:18

@Project -> File: gaussian-process-regression -> solve_optimization.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 使用高斯过程求解函数最优化
"""
import logging

logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt
import numpy as np
import random
import sys

sys.path.append('../')

from lib import proj_cmap
from lib import gen_gaussian_process_samples
from mod.gaussian_process.sampling import cal_covariance_matrix

kernel_name = 'RBF'
kernel_params = {'sigma': 10.0, 'l': 0.8}


def objective_func(t):
	return np.sin(8 * t) * np.power(t, 2) + 10 * t * np.cos(t)


if __name__ == '__main__':
	# 参数.
	x_set = np.arange(-5, 5, 0.1)
	mu = np.zeros([len(x_set), 1])
	C = cal_covariance_matrix(x_set, kernel_name, kernel_params)
	prior_samples = gen_gaussian_process_samples(x_set, mu, C, samples_n = 200)
	
	# plt.plot(xs, prior_samples, c = proj_cmap['blue'])
	
	# 观测.
	x_obs = []
	y_obs = []
	
	idxs2sample = list(range(len(x_set)))  # 通过index在x_set中进行抽样
	guess_n = 0
	while True:
		if guess_n == 0:
			idx = random.sample(idxs2sample, k = 1)[0]
		else:
			idx = np.argmax(vars)
		idxs2sample.remove(idx)  # TODO: ValueError: list.remove(x): x not in list
		
		x_new = x_set[idx]
		y_new = objective_func(x_new)
		x_obs.append(x_new)
		y_obs.append(y_new)
		obs_n = len(x_obs)
		
		# 先验参数mu.
		mu_1 = np.zeros_like(x_obs).reshape(-1, 1)
		mu_2 = np.zeros_like(x_set).reshape(-1, 1)
		
		x_total = np.hstack((np.array(x_obs), x_set))
		C = cal_covariance_matrix(x_total, kernel_name = kernel_name, kernel_params = kernel_params)
		C_11 = C[: obs_n, : obs_n]
		C_12 = C[: obs_n, obs_n:]
		C_21 = C[obs_n:, : obs_n]
		C_22 = C[obs_n:, obs_n:]
		
		# 后验参数mu和sigma.
		mu_cond = mu_2 + np.dot(np.dot(C_21, np.linalg.inv(C_11)), np.array(y_obs).reshape(-1, 1) - mu_1)
		sigma_cond = C_22 - np.dot(np.dot(C_21, np.linalg.inv(C_11)), C_12)
		
		post_samples = gen_gaussian_process_samples(x_set, mu_cond, sigma_cond, samples_n = 200)
		vars = np.var(post_samples, axis = 1)
		
		plt.clf()
		plt.scatter(x_obs, y_obs, marker = '.', s = 12, c = proj_cmap['black'])
		plt.plot(x_set, post_samples, c = proj_cmap['grey'], linewidth = 0.3)
		plt.plot(x_set, objective_func(x_set), c = proj_cmap['blue'], linewidth = 0.8)
		plt.legend(['iteration = {}'.format(guess_n)], loc = 'upper right')
		plt.show()
		plt.pause(0.3)
		
		if np.max(vars) < 1e-6:
			break
		else:
			guess_n += 1
	




