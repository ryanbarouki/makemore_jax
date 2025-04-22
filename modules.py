import jax
import jax.numpy as jnp

def sequential(layers):
    # e.g. sequential([Linear(fan_out=10), BatchNorm(), Tanh(),...])
    # return init_fun, apply_fun
    # signitures: init_fun(key, x), apply_fun(params, x, train=True)
    def init_fun(key, x):
        params = {}
        batch_stats = {}
        key, subkey = jax.random.split(key)
        fan_in = x.shape[-1]
        for i, layer in enumerate(layers):
            key, subkey = jax.random.split(subkey)
            name = layer.__class__.__name__
            fan_in, vars = layer.init(subkey, fan_in)
            if isinstance(layer, BatchNorm):
                ps, bnstats = vars
                params[f"{name}_{i}"] = ps
                batch_stats[f"{name}_{i}"] = bnstats
            elif layer.has_params:
                params[f"{name}_{i}"] = vars


        return params, batch_stats
    
    def apply_fun(params, x, train=True):
        params, batch_stats = params
        for i, layer in enumerate(layers):
            if not layer.has_params:
                x = layer(x)
                continue
            name = f"{layer.__class__.__name__}_{i}"
            ps = params[name]
            if isinstance(layer, BatchNorm):
                x, new_batch_stats = layer(ps, x, batch_stats[name], train)
                batch_stats[name] = new_batch_stats
            else:
                x = layer(ps,x)
        return x, batch_stats

    return init_fun, apply_fun

class Module:
    has_params = True

    def init(self, key, fan_in):
        return fan_in, ()

class Linear(Module):
    def __init__(self, fan_out, bias=True):
        self.fan_out = fan_out
        self.bias = bias

    def __call__(self, params, x, *args):
        W, b = params
        return x @ W + b
    
    def init(self, key, fan_in):
        W = jax.random.normal(key, (fan_in, self.fan_out)) / (fan_in**0.5)
        b = jnp.zeros(self.fan_out) if self.bias else None
        return self.fan_out, (W, b)

class BatchNorm(Module):
    def __init__(self, eps=1e-5, momentum=0.001):
        self.momentum = momentum
        self.eps = eps
    
    def __call__(self, params, x, batch_stats, train, *args):
        gamma, beta = params
        running_mean, running_std = batch_stats
        if train:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
            x = gamma * ((x-mean)/std) + beta
            running_mean = (1-self.momentum) * running_mean + self.momentum * mean
            running_std = (1-self.momentum) * running_std + self.momentum * std
        else:
            x = gamma * ((x-running_mean)/running_std) + beta
        return x, (running_mean, running_std)
    
    def init(self, key, fan_in):
        # return gamma, beta and batch_stats
        gamma = jnp.ones(fan_in)
        beta = jnp.zeros(fan_in)

        running_mean = jnp.zeros(fan_in)
        running_std = jnp.zeros(fan_in)

        return fan_in, ((gamma, beta), (running_mean, running_std))
    
class Flatten(Module):
    def __init__(self):
        self.has_params = False

    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

class Embedding(Module):
    def __init__(self, emb_dim, vocab_size):
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

    def __call__(self, params, x):
        C, = params
        return C[x]

    def init(self, key, fan_in):
        C = jax.random.normal(key, (self.vocab_size, self.emb_dim))
        return fan_in*self.emb_dim, (C,)


class EmbeddingWithFlatten(Module):
    def __init__(self, emb_dim, vocab_size):
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

    def __call__(self, params, x):
        C, = params
        emb = C[x]
        context_size = x.shape[-1]
        return emb.reshape(-1, self.emb_dim*context_size)

    def init(self, key, fan_in):
        C = jax.random.normal(key, (self.vocab_size, self.emb_dim))
        return fan_in*self.emb_dim, (C,)

class Tanh(Module):
    def __init__(self):
        self.has_params = False
    
    def __call__(self, x):
        return jnp.tanh(x)

