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
            x, vars = layer.init(subkey, x)
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

class Linear(Module):
    def __init__(self, fan_out, bias=True):
        self.fan_out = fan_out
        self.bias = bias

    def __call__(self, params, x, *args):
        if self.bias:
            W, b = params
            return x @ W + b
        W, = params
        return x @ W
    
    def init(self, key, x):
        fan_in = x.shape[-1]
        W = jax.random.normal(key, (fan_in, self.fan_out)) / (fan_in**0.5)
        b = jnp.zeros(self.fan_out) if self.bias else None
        params = (W,b) if b is not None else (W,)
        return self(params, x), params

class BatchNorm(Module):
    def __init__(self, eps=1e-5, momentum=0.001):
        self.momentum = momentum
        self.eps = eps
    
    def __call__(self, params, x, batch_stats, train, *args):
        gamma, beta = params
        running_mean, running_std = batch_stats
        if train:
            mean = x.mean(axis=0)
            var = jnp.mean((x-mean)**2, axis=0)
            std = jnp.sqrt(var + self.eps)
            x = gamma * ((x-mean)/std) + beta
            running_mean = (1-self.momentum) * running_mean + self.momentum * mean
            running_std = (1-self.momentum) * running_std + self.momentum * std
        else:
            x = gamma * ((x-running_mean)/running_std) + beta
        return x, (running_mean, running_std)
    
    def init(self, key, x):
        fan_in = x.shape[-1]
        # return gamma, beta and batch_stats
        gamma = jnp.ones_like(x)
        beta = jnp.zeros_like(x)

        running_mean = jnp.zeros_like(x)
        running_std = jnp.ones_like(x)

        params = (gamma, beta)
        batch_stats = (running_mean, running_std)
        x, _ = self(params, x, batch_stats, False)
        return x, (params, batch_stats)
    
class Flatten(Module):
    def __init__(self):
        self.has_params = False

    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

    def init(self, key, x):
        return self(x), ()
    
class FlattenConsecutive(Module):
    def __init__(self, n):
        self.has_params = False
        self.n = n

    def __call__(self, x):
        if len(x.shape) < 3:
            # to allow initialization to run on single example
            T, C = x.shape
            B = 1
        else:
            B, T, C = x.shape
        x = x.reshape(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x
    
    def init(self, key, x):
        return self(x), ()

class Embedding(Module):
    def __init__(self, emb_dim, vocab_size):
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

    def __call__(self, params, x):
        C, = params
        return C[x]

    def init(self, key, x):
        C = jax.random.normal(key, (self.vocab_size, self.emb_dim))
        params = (C,)
        return self(params, x), params

class Tanh(Module):
    def __init__(self):
        self.has_params = False
    
    def __call__(self, x):
        return jnp.tanh(x)
    
    def init(self, key, x):
        return self(x), ()

