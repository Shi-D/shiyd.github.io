# Regularization 正则化 

防止过拟合主要有两种方法:增加数据、采用正则化。但第一种往往代价较大，因此可以选择正则化来防止模型过拟合。

## L2 Regularization

将成本函数从
$$
J=−\frac{1}{m}∑_{i=1}^m(y^{(i)}log(a^{[L](i)})+(1−y^{(i)})log(1−a^{[L](i)}))
$$

变为，
$$
J_{regularized}=\underbrace{−\frac1m∑_{i=1}^m(y^{(i)}log(a^{[L](i)})+(1−y^{(i)})log(1−a^{[L](i)}))}_{cross-entropy cost}\
+\underbrace{\frac1m\fracλ2∑_l∑_k∑_jW^{[l]2}_{k,j}}_{L2 regularization cost}
$$

## Dropout

随机丢掉一些神经元