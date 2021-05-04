# Initialize 参数初始化

## zero

```python
# GRADED FUNCTION: initialize_parameters_zeros 

def initialize_parameters_zeros(layers_dims):
    
parameters = {}
L = len(layers_dims)            # number of layers in the network

for l in range(1, L):
    parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
    parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
return parameters
```


## random

```python
# GRADED FUNCTION: initialize_parameters_random

def initialize_parameters_random(layers_dims):

    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters
```



## he

```python
sqrt(2./layers_dims[l-1])
```



## Xavier

```python
sqrt(1./layers_dims[l-1])
```

