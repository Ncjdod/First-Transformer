import jax
import jax.numpy as jnp
import jax.random


def softmax(x):
    return (jnp.exp(x))/(jnp.sum(jnp.exp(x), axis=-1, keepdims=True))


def attention_layer(X, W_q, W_k, W_v):
    Q = jnp.matmul(X, W_q)
    K = jnp.matmul(X, W_k)
    V = jnp.matmul(X, W_v)
    d_k = jnp.shape(W_k)[-1]
    
    S = jnp.matmul(Q, jnp.transpose(K))
    
    mask = jnp.triu(jnp.ones_like(S, dtype=bool), k=1)
    S = jnp.where(mask, -jnp.inf, S)
   
    A = softmax((S)/jnp.sqrt(d_k))
    O = jnp.matmul(A, V)
    
    return O, A
key = jax.random.PRNGKey(42)
key1, key2, key3, key4 = jax.random.split(key, 4)
X = jax.random.normal(key1, shape=(16, 8))
W_q = jax.random.normal(key2, shape=(8, 8))
W_k = jax.random.normal(key3, shape=(8, 8))
W_v = jax.random.normal(key4, shape=(8, 8))

O, A = attention_layer(X, W_q, W_k, W_v)
print(O.shape)