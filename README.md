# gaussian-process-regression
高斯过程回归（建议使用Typora打开该文档）



#### 一、高斯分布

高斯过程（Gaussian Process, GP）是随机过程之一，是一系列符合正态分布的随机变量在一指数集（index set）内的集合。



**<u>一维高斯分布</u>**：
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



**<u>二维高斯分布</u>**：
$$
f(x,y) = A \exp (-(\frac{x - x_0)^2}{2\sigma_x^2} + \frac{(y - y_0)^2}{2\sigma_y^2}))
$$


**<u>多维高斯分布</u>**：
$$
p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \ exp (-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu))
$$
其中，$\mu$为各随机变量的均值组成的$n \times 1$向量，$\Sigma$表示随机变量间的$n \times n$协方差矩阵，正定。



#### 二、高斯过程

