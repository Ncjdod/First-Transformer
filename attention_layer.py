import jax
import jax.numpy as jnp
import jax.random


def softmax(x):
    return (jnp.exp(x))/(jnp.sum(jnp.exp(x), axis=-1, keepdims=True))


def attention_layer(X, W_q, W_k, W_v):
    def queries():
        Q = jnp.matmul(X, W_q)
        return Q
    
    def keys():
        K = jnp.matmul(X, W_k)
        return K
    
    def values():
        V = jnp.matmul(X, W_v)
        return V
    
    def s_matrix(Q=queries(), K=keys()):
        S = jnp.matmul(Q, jnp.transpose(K))
        return S
    
    def attention_matrix(S=s_matrix(), d_k=8):
        M = jnp.zeros_like(S)
        mask = jnp.triu(jnp.ones_like(S, dtype=bool), k=1)
        M = jnp.where(mask, -jnp.inf, M)
        A = softmax((S + M)/jnp.sqrt(d_k))
        return A
    
    def O(A=attention_matrix(), V=values()):
        O = jnp.matmul(A, V)
        return O
    
    return O(), attention_matrix()
key = jax.random.PRNGKey(42)
key1, key2, key3, key4 = jax.random.split(key, 4)
X = jax.random.normal(key1, shape=(4, 8))
W_q = jax.random.normal(key2, shape=(8, 8))
W_k = jax.random.normal(key3, shape=(8, 8))
W_v = jax.random.normal(key4, shape=(8, 8))

O, A = attention_layer(X, W_q, W_k, W_v)
print(A)