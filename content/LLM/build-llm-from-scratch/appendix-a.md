---
title: "Appendix A - Introduction to PyTorch"
publishedAt: "2024-11-15"
summary: "Learning the basics of pythorch."
tags:
  - LLM
  - ai
---

## What is Pytorch

It's an open source library based on Python

## The three components of Pythorch

![Screenshot 2024-11-17 122500.png](https://collected-notes.s3.us-west-2.amazonaws.com/uploads/28846/e527db7d-a441-40ad-8148-83ab625093a1)

> Figure A.1 PyTorch’s three main components include a tensor library as
> a fundamental building block for computing, automatic differentiation for
> model optimization, and deep learning utility functions, making it easier to
> implement and train deep neural network models.

- _Tensor Library:_ extends the concept of the array-oriented programming library NumPy
- _Automatic differentiation engine_: that enables the automatic computation of gradients for tensor operations, simplifying backpropagation and model optimization
- _Deep learning library:_ It offers modular, flexible, and efficient building blocks, including pretrained models, loss functions, and optimizers, for designing and training a wide range of deep learning models

## Definding deep learning

**We define a few terms for more clarity**

_AI:_ a computer program with the capability to perform human tasks like understanding natural
language, pattern recognition, make decisions.

_Machine Learning:_ a subfield of AI, Machine learning is to enable computers to learn from data and make predictions or decisions
without being explicitly programmed to perform the task
![Screenshot 2024-11-17 123617.png](https://collected-notes.s3.us-west-2.amazonaws.com/uploads/28846/979a9f76-0b29-4c08-b1c4-8f90d865fefb)

_Deep Learning:_ is a subcategory of machine learning that focuses on the training and
application of deep neural networks. The “deep” in deep learning refers to the multiple hidden layers of artificial neurons or nodes that allow them to model complex, nonlinear relationships
in the data. Unlike traditional machine learning techniques that excel at simple pattern recognition, deep learning is particularly good at handling unstructured data like images, audio, or text, so it is particularly well suited for LLMs.

![ec55ad04-11ca-44f3-8eec-9399936c26ff.webp](https://collected-notes.s3.us-west-2.amazonaws.com/uploads/28846/2dd6c9d4-9f60-4093-837f-a405afdb9d55)

## Understanding Tensors

Tensors represent a mathematical concept that generalizes vectors and matrices to potentially higher dimensions.
Tensors can be characterized by order (or rank) that provides the number of dimensions.

![Screenshot 2024-11-18 130420.png](https://collected-notes.s3.us-west-2.amazonaws.com/uploads/28846/50dea009-2c1e-43a3-a813-d71e41382d21)

From a computational perspective, tensors serve as data containers. For instance, they
hold multidimensional data, where each dimension represents a different feature.

## Scalars, vectors, matrices and tensors

```python
# zero-dimensional (scalar) tensor
tensor0d = torch.tensor(1)

# one-dimensional tensor
tensor1d = torch.tensor([1, 2, 3])

# two-dimensional tensor
tensor2d = torch.tensor([[1, 2, 3], [1, 2, 3]])

# three-dimensional tensor
tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

## Tensors Data Types

PyTorch adopts the default 64-bit integer data type from Python. If we create tensors from Python floats, PyTorch creates tensors with a 32-bit precision.

```python
tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)
# torch.int64

floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)
# torch.float32
```

> A 32-bit floating-point number offers sufficient precision for most deep learning
> tasks while consuming less memory and computational resources than a 64-bit floatingpoint number. Moreover, GPU architectures are optimized for 32-bit computations, and
> using this data type can significantly speed up model training and inference.

## Implementing multilayer neural networks

![Screenshot 2024-11-18 141315.png](https://collected-notes.s3.us-west-2.amazonaws.com/uploads/28846/a20bc460-1935-4317-91aa-d9d5a5487d37)

```python
# classic multilayer perceptron with two hidden layers
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            # The Linear layer takes the number of input and output nodes as arguments.
            torch.nn.Linear(num_inputs, 30),
            # Nonlinear activation functions are placed between the hidden layers.
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


model = NeuralNetwork(50, 3)
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

print(model.layers[0].weight)
print(model.layers[0].weight.shape)
```

## Setting up efficient data loaders

![Screenshot 2024-11-18 160240.png](https://collected-notes.s3.us-west-2.amazonaws.com/uploads/28846/5dc8e284-fbd5-41b2-9716-045bf632ec8d)

We create a dataset and the test-data with 5 training examples, each with two features. Then, we create a tensor with the labels.

```python
# Custom dataset class
# Creating a small toy dataset
X_train = torch.tensor(
    [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
)
Y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor(
    [
        [-0.8, 2.8],
        [2.6, -1.6],
    ]
)
Y_test = torch.Tensor([0, 1])


# Defining a custom Dataset class
class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.features = x
        self.labels = y

    # Instructions for retrieving exactly one data record and the corresponding label
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


train_ds = ToyDataset(X_train, Y_train)
test_ds = ToyDataset(X_test, Y_test)

print(len(train_ds))
# 5
```

Now that we’ve defined a PyTorch Dataset class we can use for our toy dataset, we can
use PyTorch’s _DataLoader_ class to sample from it.

```python
torch.manual_seed(123)

train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_ds, batch_size=2, shuffle=True, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch: {idx+1}:", x, y)

```

- dataset: this provides the data set
- shuffle: if the data is mixed
- batch_size: numbers of samples
- num_workers: specifies how many parallel processes should load the data

![Screenshot 2024-11-18 165752.png](https://collected-notes.s3.us-west-2.amazonaws.com/uploads/28846/dcde0d92-28ab-4e83-8f0f-7ec4d20ddd33)
