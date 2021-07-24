# Notation

$$
W^{[layer]}_{unit}
\text{, of shape }
(n^{[layer]} \times n^{[layer - 1]})
$$

$$
X^{(example)}_{feature}
\text{, of shape }
(n_x \times m)
$$

$$
X = A^{[0]} = 
\left[
  \begin{array}{ccc}
	| & | &  & | \\
	| & | &  & | \\
	X^{(1)} & X^{(2)} & \dots & X^{(m)} \\
	| & | &  & | \\
	| & | &  & | \\
  \end{array}
\right]
,
X \ni \Reals^{n_x \times m}
$$

# Forward
$$
Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}
$$

$$ A^{[l]} = \sigma^{[l]} (Z^{[l]}) $$

While keeping numpy broadcast in mind,
here are the dimensions of the different matrix/vectors

$$
\underset{n^{[1]} \times m}{Z^{[1]}}
= 
\underset{n^{[1]} \times n^{[0]}}{W^{[1]}}
\cdot
\underset{n^{[0]}\times m}{A^{[0]}}
+
\underset{n^{[1]}\times 1}{b^{[1]}}
$$


$$
\def\horzbar{\text{--------}}
Z^{[l]} = 
\left[
  \begin{array}{ccc}
    \horzbar & w^{[l]T}_{1} & \horzbar \\
    \horzbar & w^{[l]T}_{2} & \horzbar \\
    \horzbar & w^{[l]T}_{3} & \horzbar \\
    \horzbar & w^{[l]T}_{4} & \horzbar \\  
  \end{array}
\right]
\cdot
\begin{bmatrix}
	x_{1}^{(1)} & | &  & | & \\
	x_{2}^{(1)} & x^{(2)} & \dots & x^{(m)} & \\
	x_{3}^{(1)} & | &  & | & \\
	x_{4}^{(1)} & | &  & | & \\
\end{bmatrix}
+
\left[
  \begin{array}{ccc}
    b^{[l]}_{1} \\
    b^{[l]}_{2} \\
    b^{[l]}_{3} \\
    b^{[l]}_{4} \\
  \end{array}
\right] = 
\left[
  \begin{array}{ccc}
	| & | &  & | \\
	| & | &  & | \\
	Z^{[l](1)} & Z^{[l](2)} & \dots & Z^{[l](m)} \\
	| & | &  & | \\
	| & | &  & | \\
  \end{array}
\right]

$$


# Backpropagation

$\delta^l \equiv \delta A^l$

| Definition                       | Equation |
|:-                                |-:|
|Output layer error                |${\color{green}\delta^L} = \nabla_a C \odot \sigma'(z^L)$|
|layer error                       |$\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$|
|Cost partial derivate for Bias    |$\frac{\partial C}{\partial b^l_j} =  \delta^l_j$|
|Cost partial derivate for Weights |$\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j$|

## 1. The error in the output layer $δ^L$

$$
\delta^L_j = \frac{\partial C}{\partial a^L_j} \sigma'(z^L_j)
$$

Matrix based version:
$$
\delta^L = \nabla_a C \odot \sigma'(z^L)
$$

|Name|$C$|$\nabla_a C$|
|:-|:-|:-|
|Mean Squared Error|$\frac{1}{2}∑_j(y_j−a^L_j)^2$|$a^L-y$|
|Binary Cross Entropy|$y\log{a^L} + (1-y)\log(1-a^L)$|$\frac{-y}{a^L}+\frac{1-y}{1-a^L}$|
|SoftMax|$\sigma(\vec{z})_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{n^L} e^{z_{j}}}$|$$|
||$$|$$|

SoftMax derivation

$$
\sigma(\vec{z})_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{n^L} e^{z_{j}}}
$$


$$
\frac{\partial L}{\partial o_i}=-\sum_ky_k\frac{\partial \log p_k}{\partial o_i}=-\sum_ky_k\frac{1}{p_k}\frac{\partial p_k}{\partial o_i}\\=-y_i(1-p_i)-\sum_{k\neq i}y_k\frac{1}{p_k}({\color{red}{-p_kp_i}})\\=-y_i(1-p_i)+\sum_{k\neq i}y_k({\color{red}{p_i}})\\=-y_i+\color{blue}{y_ip_i+\sum_{k\neq i}y_k({p_i})}\\=\color{blue}{p_i\left(\sum_ky_k\right)}-y_i=p_i-y_i
$$

For example if we use the quadratic cost function
$$
C=\frac{1}{2}∑_j(y_j−a^L_j)^2
$$

It's derivate relative to $a^L_j$ will be :
$$
\frac{∂C}{∂a^L_j}=(a^L_j−y_j)
$$

And it's vectorized form:
$$
\nabla_a C = (a^L-y)
$$

## 2. The error $δ^l$ in terms of the error in the next layer

$$
\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)
$$

## 3. Rate of change of the cost with respect to any bias

$$
\frac{\partial C}{\partial b^l_j} =  \delta^l_j
$$

## 4. Rate of change of the cost with respect to any weight

$$
  \frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j
$$

Can be seen as this simplified form:

$$
\frac{\partial C}{\partial w} = a_{\rm in} \delta_{\rm out}
$$

# Optimizers
Let's write our weight update in the following way:
$$
\delta W \equiv \frac{\partial C}{\partial W}
$$
The following optimizers equations works similarly for $\frac{\partial C}{\partial b}$

## Momentum

$V\delta W$ should be initialized at $0$
$$
V\delta W = (\beta) V\delta W + (1 - \beta) \delta W  
\newline
W = W - \alpha \times V\delta W
$$

## RMSprop

$S\delta W$ should be initialized at $0$
$$
S\delta W = (\beta) S\delta W + (1 - \beta) \delta W^2
\newline
W = W - \alpha \times \frac{\delta W}{\sqrt{S\delta W} + \epsilon}
$$

## Adam


### 1. Computing the Momentum and RMSprop
$V\delta W$ and $S\delta W$ should be initialized at $0$

$$
V\delta W = (\beta_1) V\delta W + (1 - \beta_1) \delta W  
\newline
S\delta W = (\beta_2) S\delta W + (1 - \beta_2) \delta W^2
$$

### 2. Correcting exponentially weighted averages
$t$ is the umpteenth update
$$
V\delta W^{corrected} = \frac{V \delta W}{1 - \beta^t_1}
\newline
S\delta W^{corrected} = \frac{S \delta W}{1 - \beta^t_2}
$$

### 3. Weight update

$$
W = W - \alpha \times \frac{V\delta W^{corrected}}{\sqrt{S\delta W^{corrected}} + \epsilon}
\newline
$$

| HyperParameter | Advised Value |
|:-|-:|
|$\alpha$|needs to be tuned|
|$\beta_1$|$0.9$|
|$\beta_2$|$0.999$|
|$\epsilon$|$10^{-8}$|