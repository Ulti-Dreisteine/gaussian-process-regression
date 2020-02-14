# -*- coding: utf-8 -*-
"""
Created on 2020/2/14 17:28

@Project -> File: gaussian-process-regression -> regression.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 过程回归
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
kernel_params = {'sigma': 10.0, 'l': 0.5}


def cal_prior_metrics(x_obs_list, x_ticks):
	"""
	计算先验分布参数
	:param x_obs_list: list, 观测到的x
	:param x_ticks: list, 回归设定的x_ticks
	"""
	# 1表示观测值, 2表示设定ticks.
	obs_n = len(x_obs_list)
	mu_1 = np.zeros_like(x_obs_list).reshape(-1, 1)
	mu_2 = np.zeros_like(x_ticks).reshape(-1, 1)
	
	x_total = np.hstack((np.array(x_obs_list), x_ticks))
	C = cal_covariance_matrix(x_total, kernel_name = kernel_name, kernel_params = kernel_params)
	C_11 = C[: obs_n, : obs_n]
	C_12 = C[: obs_n, obs_n:]
	C_21 = C[obs_n:, : obs_n]
	C_22 = C[obs_n:, obs_n:]
	
	return mu_1, mu_2, C_11, C_12, C_21, C_22


def cal_posterior_metrics(y_obs_list, mu_1, mu_2, C_11, C_12, C_21, C_22):
	"""计算后验分布参数"""
	mu_cond = mu_2 + np.dot(np.dot(C_21, np.linalg.inv(C_11)), np.array(y_obs_list).reshape(-1, 1) - mu_1)
	sigma_cond = C_22 - np.dot(np.dot(C_21, np.linalg.inv(C_11)), C_12)
	return mu_cond, sigma_cond


class UnivariateRegression(object):
	"""一维函数回归"""
	
	def __init__(self, x_ticks):
		"""
		初始化
		:param x_ticks: array like, 用于计算回归结果的x序列
		"""
		x_ticks = np.array(x_ticks).flatten()
		self.x_ticks = x_ticks
	
	def _gen_idxs2sample(self):
		idxs2sample = list(range(len(self.x_ticks)))
		return idxs2sample
	
	def regress(self, func, tol = 1e-2, show_detail = False, verbose = False):
		"""
		执行函数回归
		:param func: func object, 目标函数
		:param tol: float, 方差阈值
		
		Example:
		------------------------------------------------------------
		def objective_func(t):
			return np.sin(8 * t) * np.power(t, 2)
	
	
		x_ticks = np.arange(-10, 10, 0.1)
		ur = UnivariateRegression(x_ticks)
		mu_cond, sigma_cond = ur.regress(objective_func, show_detail = True, verbose = True)
		------------------------------------------------------------
		"""
		# 初始化抽样index集.
		idxs2sample = self._gen_idxs2sample()
		
		# 初始化观测集.
		x_obs_list, y_obs_list = [], []
		
		iter_n = 0
		while True:
			# 抽样.
			if iter_n == 0:
				idx_chosen = random.sample(idxs2sample, k = 1)[0]
			else:
				idxes = np.argsort(variances)[::-1]  # 从大到小排序 TODO: 目前算法不支持对同一x的多次采样，否则会报错
				idx_chosen = None
				for idx in idxes:
					if idx in idxs2sample:
						idx_chosen = idx
						break
			
			idxs2sample.remove(idx_chosen)
			
			# 观测.
			x_obs = self.x_ticks[idx_chosen]
			y_obs = func(x_obs)
			x_obs_list.append(x_obs)
			y_obs_list.append(y_obs)
			
			# 计算先验参数.
			mu_1, mu_2, C_11, C_12, C_21, C_22 = cal_prior_metrics(x_obs_list, self.x_ticks)
			
			# 计算后验参数.
			mu_cond, sigma_cond = cal_posterior_metrics(y_obs_list, mu_1, mu_2, C_11, C_12, C_21, C_22)
			
			# 后验分布采样.
			post_samples = gen_gaussian_process_samples(self.x_ticks, mu_cond, sigma_cond, samples_n = 200)
			variances = np.var(post_samples, axis = 1)
			
			if show_detail:
				plt.clf()
				plt.scatter(x_obs_list, y_obs_list, marker = '.', s = 12, c = proj_cmap['black'])
				plt.plot(self.x_ticks, post_samples, c = proj_cmap['grey'], linewidth = 0.3)
				plt.plot(self.x_ticks, func(self.x_ticks), c = proj_cmap['blue'], linewidth = 0.8)
				plt.legend(['iteration = {}'.format(iter_n)], loc = 'upper right')
				plt.show()
				plt.pause(0.3)
				
			if verbose:
				print('step: {}		max variance: {:.6f}.'.format(iter_n, np.max(variances)))
			
			if np.max(variances) < tol:
				break
			elif len(idxs2sample) == 0:
				break
			else:
				iter_n += 1
		
		return mu_cond, sigma_cond
	
	




