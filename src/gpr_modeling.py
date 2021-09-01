# -*- coding: utf-8 -*-
"""
Created on 2021/09/01 10:33:40

@File -> gpr_modeling.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 高斯过程回归建模
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../' * 2))
sys.path.append(BASE_DIR)

from src.settings import PROJ_DIR, plt
from core.gaussian_process.sampling import calCovMatrix, genGaussianProcessSamples

if __name__ == '__main__':
    t_series = np.arange(0, 10, 0.1)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # ---- 先验分布 ---------------------------------------------------------------------------------

    mu = np.zeros([t_series.shape[0], 1])
    C = calCovMatrix(t_series, kernel_name='RBF')

    samples_prior = genGaussianProcessSamples(t_series, mu, C, samples_n=500)

    # 画图.
    ax = axs[0]
    for i in range(samples_prior.shape[1]):
        ax.plot(t_series, samples_prior[:, i], c='grey', alpha=0.1)
    ax.plot(t_series, np.mean(samples_prior, axis=1), c='b')
    ax.set_ylabel('$y$')
    ax.set_title('Prior Distribution', fontsize = 15)

    # ---- 采集后验样本 -----------------------------------------------------------------------------

    # 采集后验样本.
    t_obs = np.array([1.1, 1.0, 4.0, 6.0, 7.0, 7.5])
    x_obs = np.array([1.0, 1.0, 0.5, 1.0, 2.0, 2.0])
    N_obs = x_obs.shape[0]

    # 计算先验分布参数.
    t_total = np.hstack((t_obs, t_series))
    mu_prioir = np.zeros_like(t_total)
    C_prior = calCovMatrix(t_total, kernel_name='RBF')

    # 更新后验分布参数, 1代表观测值, 2代表未知值.
    mu_1, mu_2 = mu_prioir[:N_obs].reshape(-1, 1), mu_prioir[N_obs:].reshape(-1, 1)
    C_11 = C_prior[:N_obs, :N_obs]
    C_12 = C_prior[:N_obs, N_obs:]
    C_21 = C_prior[N_obs:, :N_obs]
    C_22 = C_prior[N_obs:, N_obs:]
    mu_post = mu_2 + np.dot(np.dot(C_21, np.linalg.inv(C_11)), x_obs.reshape(-1, 1) - mu_1)
    sigma_post = C_22 - np.dot(np.dot(C_21, np.linalg.inv(C_11)), C_12)

    samples_post = genGaussianProcessSamples(t_series, mu_post, sigma_post, samples_n=500)

    # 画图.
    ax = axs[1]
    for i in range(samples_post.shape[1]):
        ax.plot(t_series, samples_post[:, i], c='grey', alpha=0.1, zorder = -i)
    for i in range(N_obs):
        ax.scatter(t_obs[i], x_obs[i], s = 60, c = 'k')
    ax.plot(t_series, np.mean(samples_post, axis=1), c='b', zorder=0)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$y$')
    ax.set_title('Posterior Distribution', fontsize = 15)

    fig.tight_layout()
    fig.savefig(os.path.join(PROJ_DIR, 'img/prior_vs_posterior.png'), dpi = 450)
