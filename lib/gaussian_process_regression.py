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
import seaborn as sns
import numpy as np
import sys
import os

sys.path.append('../')

from lib import proj_dir, proj_cmap
from lib import gen_gaussian_process_samples
from mod.gaussian_process.sampling import cal_covariance_matrix

kernel_name = 'RBF'
kernel_params = {'sigma': 0.5, 'l': 1.0, 'p': 0.5}

# kernel_name = 'periodic'
# kernel_params = {'sigma': 0.5, 'l': 1.0, 'p': 0.5}

# kernel_name = 'linear'
# kernel_params = {'sigma': 0.5, 'c': 0.5, 'sigma_b': 0.5}


if __name__ == '__main__':
	# 生成一批先验样本.
	t_list = np.arange(0, 10, 0.1)
	mu = np.zeros([len(t_list), 1])
	C = cal_covariance_matrix(t_list, kernel_name = kernel_name, kernel_params = kernel_params)
	sns.heatmap(C, cmap = 'Blues')
	prior_samples = gen_gaussian_process_samples(t_list, mu, C, samples_n = 200)
	
	plt.figure('Gaussian Process Regression')
	plt.suptitle('Gaussian Process Regression')
	plt.subplot(2, 1, 1)
	plt.plot(t_list, prior_samples, c = proj_cmap['grey'], linewidth = 0.3, alpha = 0.3)
	plt.plot(t_list, np.mean(prior_samples, axis = 1), c = proj_cmap['blue'], linewidth = 2.0, alpha = 0.6)
	plt.legend(['Prior'], loc = 'upper right', fontsize = 8)
	plt.xlim([np.min(t_list), np.max(t_list)])
	plt.xticks(fontsize = 6)
	plt.yticks(fontsize = 6)

	# 采集后验样本.
	t_obs = [1.1, 1.0, 4.0, 6.0, 7.0, 7.1]
	x_obs = [1.0, 1.0, 0.5, 1.0, 2.0, 2.0]
	obs_n = len(x_obs)

	# %% 计算先验和后验分布参数, 1代表观测值, 2代表未知值.
	# 先验参数mu.
	mu_1 = np.zeros_like(t_obs).reshape(-1, 1)
	mu_2 = np.zeros_like(t_list).reshape(-1, 1)

	# 先验参数Sigma.
	t_total = np.hstack((np.array(t_obs), t_list))
	C = cal_covariance_matrix(t_total, kernel_name = kernel_name, kernel_params = kernel_params)
	C_11 = C[: obs_n, : obs_n]
	C_12 = C[: obs_n, obs_n:]
	C_21 = C[obs_n:, : obs_n]
	C_22 = C[obs_n:, obs_n:]

	# 后验参数mu和sigma.
	mu_cond = mu_2 + np.dot(np.dot(C_21, np.linalg.inv(C_11)), np.array(x_obs).reshape(-1, 1) - mu_1)
	sigma_cond = C_22 - np.dot(np.dot(C_21, np.linalg.inv(C_11)), C_12)

	# 生成后验分布样本.
	post_samples = gen_gaussian_process_samples(t_list, mu_cond, sigma_cond, samples_n = 200)
	plt.subplot(2, 1, 2)
	plt.plot(t_list, post_samples, c = proj_cmap['grey'], linewidth = 0.3, alpha = 0.3)
	plt.plot(t_list, np.mean(post_samples, axis = 1), c = proj_cmap['blue'], linewidth = 2.0, alpha = 0.6)
	plt.scatter(t_obs, x_obs, marker = 'o', c = proj_cmap['black'])
	plt.legend(['Posterior'], loc = 'upper right', fontsize = 8)
	plt.xlim([np.min(t_list), np.max(t_list)])
	plt.xticks(fontsize = 6)
	plt.yticks(fontsize = 6)
	plt.xlabel('index', fontsize = 10)

	plt.savefig(os.path.join(proj_dir, 'graph/prior_vs_posterior.png'), dpi = 450)
	
	
	


