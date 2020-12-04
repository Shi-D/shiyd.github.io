# class 1

PyTorch是一个基于Python的科学计算库，它有以下特点:

- 类似于NumPy，但是它可以使用GPU
- 可以用它定义深度学习模型，可以灵活地进行深度学习模型的训练和使用

## Tensors

Tensor类似与NumPy的ndarray，唯一的区别是Tensor可以在GPU上加速运算。

In [1]:

```
from __future__ import print_function
import torch
```

构造一个未初始化的5x3矩阵:

In [2]:

```
x = torch.empty(5, 3)
print(x)
tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 4.7339e+30, 1.4347e-19],
        [2.7909e+23, 1.8037e+28, 1.7237e+25],
        [9.1041e-12, 6.2609e+22, 4.7428e+30],
        [3.8001e-39, 0.0000e+00, 0.0000e+00]])
```

构建一个随机初始化的矩阵:

In [3]:

```
x = torch.rand(5, 3)
print(x)
tensor([[0.4821, 0.3854, 0.8517],
        [0.7962, 0.0632, 0.5409],
        [0.8891, 0.6112, 0.7829],
        [0.0715, 0.8069, 0.2608],
        [0.3292, 0.0119, 0.2759]])
```

构建一个全部为0，类型为long的矩阵:

In [4]:

```
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```

从数据直接直接构建tensor:

In [5]:

```
x = torch.tensor([5.5, 3])
print(x)
tensor([5.5000, 3.0000])
```

也可以从一个已有的tensor构建一个tensor。这些方法会重用原来tensor的特征，例如，数据类型，除非提供新的数据。

In [6]:

```
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[ 1.4793, -2.4772,  0.9738],
        [ 2.0328,  1.3981,  1.7509],
        [-0.7931, -0.0291, -0.6803],
        [-1.2944, -0.7352, -0.9346],
        [ 0.5917, -0.5149, -1.8149]])
```

得到tensor的形状:

In [7]:

```
print(x.size())
torch.Size([5, 3])
```

#### 注意

``torch.Size`` 返回的是一个tuple

Operations

有很多种tensor运算。我们先介绍加法运算。

In [8]:

```
y = torch.rand(5, 3)
print(x + y)
tensor([[ 1.7113, -1.5490,  1.4009],
        [ 2.4590,  1.6504,  2.6889],
        [-0.3609,  0.4950, -0.3357],
        [-0.5029, -0.3086, -0.1498],
        [ 1.2850, -0.3189, -0.8868]])
```

另一种着加法的写法

In [9]:

```
print(torch.add(x, y))
tensor([[ 1.7113, -1.5490,  1.4009],
        [ 2.4590,  1.6504,  2.6889],
        [-0.3609,  0.4950, -0.3357],
        [-0.5029, -0.3086, -0.1498],
        [ 1.2850, -0.3189, -0.8868]])
```

加法：把输出作为一个变量

In [10]:

```
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
tensor([[ 1.7113, -1.5490,  1.4009],
        [ 2.4590,  1.6504,  2.6889],
        [-0.3609,  0.4950, -0.3357],
        [-0.5029, -0.3086, -0.1498],
        [ 1.2850, -0.3189, -0.8868]])
```

in-place加法

In [11]:

```
# adds x to y
y.add_(x)
print(y)
tensor([[ 1.7113, -1.5490,  1.4009],
        [ 2.4590,  1.6504,  2.6889],
        [-0.3609,  0.4950, -0.3357],
        [-0.5029, -0.3086, -0.1498],
        [ 1.2850, -0.3189, -0.8868]])
```

#### 注意

任何in-place的运算都会以``_``结尾。 举例来说：``x.copy_(y)``, ``x.t_()``, 会改变 ``x``。

各种类似NumPy的indexing都可以在PyTorch tensor上面使用。

In [12]:

```
print(x[:, 1])
tensor([-2.4772,  1.3981, -0.0291, -0.7352, -0.5149])
```

Resizing: 如果你希望resize/reshape一个tensor，可以使用`torch.view`：

In [13]:

```
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

如果你有一个只有一个元素的tensor，使用`.item()`方法可以把里面的value变成Python数值。

In [14]:

```
x = torch.randn(1)
print(x)
print(x.item())
tensor([0.4726])
0.4726296067237854
```

**更多阅读**

各种Tensor operations, 包括transposing, indexing, slicing, mathematical operations, linear algebra, random numbers在 `<https://pytorch.org/docs/torch>`.

## Numpy和Tensor之间的转化

在Torch Tensor和NumPy array之间相互转化非常容易。

Torch Tensor和NumPy array会共享内存，所以改变其中一项也会改变另一项。

把Torch Tensor转变成NumPy Array

In [15]:

```
a = torch.ones(5)
print(a)
tensor([1., 1., 1., 1., 1.])
```

In [16]:

```
b = a.numpy()
print(b)
[1. 1. 1. 1. 1.]
```

改变numpy array里面的值。

In [17]:

```
a.add_(1)
print(a)
print(b)
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```

把NumPy ndarray转成Torch Tensor

In [18]:

```
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

所有CPU上的Tensor都支持转成numpy或者从numpy转成Tensor。

## CUDA Tensors

使用`.to`方法，Tensor可以被移动到别的device上。

In [19]:

```
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
```

## 热身: 用numpy实现两层神经网络

一个全连接ReLU神经网络，一个隐藏层，没有bias。用来从x预测y，使用L2 Loss。

这一实现完全使用numpy来计算前向神经网络，loss，和反向传播。

numpy ndarray是一个普通的n维array。它不知道任何关于深度学习或者梯度(gradient)的知识，也不知道计算图(computation graph)，只是一种用来计算数学运算的数据结构。

In [20]:

```
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    
    # loss = (y_pred - y) ** 2
    grad_y_pred = 2.0 * (y_pred - y)
    # 
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

## PyTorch: Tensors

这次我们使用PyTorch tensors来创建前向神经网络，计算损失，以及反向传播。

一个PyTorch Tensor很像一个numpy的ndarray。但是它和numpy ndarray最大的区别是，PyTorch Tensor可以在CPU或者GPU上运算。如果想要在GPU上运算，就需要把Tensor换成cuda类型。

In [21]:

```
import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

```

简单的autograd

In [22]:

```
# Create tensors.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1
tensor(2.)
tensor(1.)
tensor(1.)
```

## PyTorch: Tensor和autograd

PyTorch的一个重要功能就是autograd，也就是说只要定义了forward pass(前向神经网络)，计算了loss之后，PyTorch可以自动求导计算模型所有参数的梯度。

一个PyTorch的Tensor表示计算图中的一个节点。如果`x`是一个Tensor并且`x.requires_grad=True`那么`x.grad`是另一个储存着`x`当前梯度(相对于一个scalar，常常是loss)的向量。

In [23]:

```
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N 是 batch size; D_in 是 input dimension;
# H 是 hidden dimension; D_out 是 output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机的Tensor来保存输入和输出
# 设定requires_grad=False表示在反向传播的时候我们不需要计算gradient
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 创建随机的Tensor和权重。
# 设置requires_grad=True表示我们希望反向传播的时候计算Tensor的gradient
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 前向传播:通过Tensor预测y；这个和普通的神经网络的前向传播没有任何不同，
    # 但是我们不需要保存网络的中间运算结果，因为我们不需要手动计算反向传播。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 通过前向传播计算loss
    # loss是一个形状为(1，)的Tensor
    # loss.item()可以给我们返回一个loss的scalar
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # PyTorch给我们提供了autograd的方法做反向传播。如果一个Tensor的requires_grad=True，
    # backward会自动计算loss相对于每个Tensor的gradient。在backward之后，
    # w1.grad和w2.grad会包含两个loss相对于两个Tensor的gradient信息。
    loss.backward()

    # 我们可以手动做gradient descent(后面我们会介绍自动的方法)。
    # 用torch.no_grad()包含以下statements，因为w1和w2都是requires_grad=True，
    # 但是在更新weights之后我们并不需要再做autograd。
    # 另一种方法是在weight.data和weight.grad.data上做操作，这样就不会对grad产生影响。
    # tensor.data会我们一个tensor，这个tensor和原来的tensor指向相同的内存空间，
    # 但是不会记录计算图的历史。
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
```

## PyTorch: nn

这次我们使用PyTorch中nn这个库来构建网络。 用PyTorch autograd来构建计算图和计算gradients， 然后PyTorch会帮我们自动计算gradient。

In [24]:

```
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
```

## PyTorch: optim

这一次我们不再手动更新模型的weights,而是使用optim这个包来帮助我们更新参数。 optim这个package提供了各种不同的模型优化方法，包括SGD+momentum, RMSProp, Adam等等。

In [25]:

```
import torch
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
```

## PyTorch: 自定义 nn Modules

我们可以定义一个模型，这个模型继承自nn.Module类。如果需要定义一个比Sequential模型更加复杂的模型，就需要定义nn.Module模型。

In [26]:

```
import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

# FizzBuzz

FizzBuzz是一个简单的小游戏。游戏规则如下：从1开始往上数数，当遇到3的倍数的时候，说fizz，当遇到5的倍数，说buzz，当遇到15的倍数，就说fizzbuzz，其他情况下则正常数数。

我们可以写一个简单的小程序来决定要返回正常数值还是fizz, buzz 或者 fizzbuzz。

In [27]:

```
# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else:             return 0
    
def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

print(fizz_buzz_decode(1, fizz_buzz_encode(1)))
print(fizz_buzz_decode(2, fizz_buzz_encode(2)))
print(fizz_buzz_decode(5, fizz_buzz_encode(5)))
print(fizz_buzz_decode(12, fizz_buzz_encode(12)))
print(fizz_buzz_decode(15, fizz_buzz_encode(15)))
1
2
buzz
fizz
fizzbuzz
```

我们首先定义模型的输入与输出(训练数据)

In [28]:

```
import numpy as np
import torch

NUM_DIGITS = 10

# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])
```

然后我们用PyTorch定义模型

In [29]:

```
# Define the model
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
)
```

- 为了让我们的模型学会FizzBuzz这个游戏，我们需要定义一个损失函数，和一个优化算法。
- 这个优化算法会不断优化（降低）损失函数，使得模型的在该任务上取得尽可能低的损失值。
- 损失值低往往表示我们的模型表现好，损失值高表示我们的模型表现差。
- 由于FizzBuzz游戏本质上是一个分类问题，我们选用Cross Entropyy Loss函数。
- 优化函数我们选用Stochastic Gradient Descent。

In [30]:

```
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)
```

以下是模型的训练代码

In [31]:

```
# Start training it
BATCH_SIZE = 128
for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Find loss on training data
    loss = loss_fn(model(trX), trY).item()
    print('Epoch:', epoch, 'Loss:', loss)
```

最后我们用训练好的模型尝试在1到100这些数字上玩FizzBuzz游戏

In [ ]:

```
# Output now
testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
with torch.no_grad():
    testY = model(testX)
predictions = zip(range(1, 101), list(testY.max(1)[1].data.tolist()))

print([fizz_buzz_decode(i, x) for (i, x) in predictions])
```

In [ ]:

```
print(np.sum(testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1,101)])))
testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1,101)])
```