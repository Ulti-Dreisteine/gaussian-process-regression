# gaussian-process-regression

高斯过程回归（建议使用Typora打开该文档）



#### 一、高斯分布

高斯过程（Gaussian Process, GP）是随机过程之一，是一系列符合正态分布的随机变量在一指数集（index set）内的集合。



**<u>一元高斯分布</u>**：
$$
X \sim N(\mu, \sigma^2)
$$
其概率密度函数为：
$$
f(x) = \frac{1}{\sigma \sqrt{2 \pi}}e^{{-(x - \mu)^2} / ({2 \sigma^2})}
$$
标准正态分布：
$$
\mu = 0, \sigma = 1
$$
正态分布具有如下性质：

1. 如果$X \sim N(\mu, \sigma^2)$，且$a$、$b$均为实数，则$aX + b \sim N(a \mu + b, (a \sigma)^2)$；

2. 如果$X \sim N(\mu_x, \sigma_x^2)$与$Y \sim N(\mu_y, \sigma_y^2)$独立，则：

   1. $U = X + Y \sim N(\mu_x + \mu_y, \sigma_x^2 + \sigma_y^2)$；
   2. $V = X - Y \sim N(\mu_x - \mu_y, \sigma_x^2 + \sigma_y^2)$；

3. 若以上$X$与$Y$相互独立，则：

   1. $XY$符合以下概率密度分布：
      $$
      p(z)=\frac{1}{\pi \sigma_x \sigma_y}K_0(\frac{|z|}{\sigma_x \sigma_y})
      $$
      其中$K_0$为修正贝塞尔函数；

   2. $X/Y$符合柯西分布：
      $$
      X/Y \sim {\rm Cauchy}(0, \sigma_x / \sigma_y)
      $$

4. 若$X_1, ..., X_n$各自独立，符合正态分布，则$X_1^2 + X_2^2 + ... + X_n^2$符合自由度为$n$的卡方分布；



**<u>二元高斯分布</u>**：
$$
f(x,y) = A \exp (-(\frac{x - x_0)^2}{2\sigma_x^2} + \frac{(y - y_0)^2}{2\sigma_y^2}))
$$

**<u>多元高斯分布</u>**：
$$
p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \ \exp(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu))
$$
其中，$\mu$为各随机变量的均值组成的$n \times 1$向量，$\Sigma$表示随机变量间的$n \times n$协方差矩阵，正定。



#### 二、多元高斯分布的条件概率密度

令随机向量$X = [x_1, x_2, ..., x_n]$服从多元高斯分布$X \sim N(\mu, \Sigma)$，令$X_1 = [x_1, ..., x_m]$为已经观测变量，$X_2 = [x_{m+1}, ..., x_n]$为未知变量，则：
$$
\begin{equation}
X = \left(
	\begin{array}{c}
	X_1 \\
	X_2
	\end{array}
\right)
\end{equation}
$$
从而有：
$$
\begin{equation}
\mu = \left(
	\begin{array}{c}
	\mu_1 \\
	\mu_2
	\end{array}
\right)
\end{equation}
$$

$$
\begin{equation}
\Sigma = \left[
	\begin{array}{}
	\Sigma_{11}, &\Sigma_{12} \\
	\Sigma_{21}, &\Sigma_{22}
	\end{array}
\right]
\end{equation}
$$

给定$X_1$求$X_2$的分布（这部分推导应该可以从相关文献中查到）：
$$
\mu_{2|1} = \mu_2 + \Sigma_{21} \Sigma_{11}^{-1}(X_1 - \mu_1)
$$

$$
\Sigma_{2|1} = \Sigma_{22} - \Sigma_{21} \Sigma_{11}^{-1} \Sigma_{12}
$$



#### 三、高斯过程回归

是一系列服从正态分布的随机变量在一指数集（index set）内的组合。