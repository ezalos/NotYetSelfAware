# Backpropagation

## 1. The error in the output layer $δ^L$

$$
\delta^L_j = \frac{\partial C}{\partial a^L_j} \sigma'(z^L_j)
$$

Matrix based version:
$$
\delta^L = \nabla_a C \odot \sigma'(z^L)
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