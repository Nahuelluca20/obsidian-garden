---
title: "ch4 - Implementing a GPT model from scratch to generate text"
tags:
  - ai
  - llm
publishedAt: "2025-03-05"
summary: "Implementing a GPT model from scratch to generate text"
---

![[Pasted image 20250305102130.png]]

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}
```

- _vocab_size_ refers to a vocabulary of 50,257 words, as used by the BPE tokenizer.
- _vocab_size_ refers to a vocabulary of 50,257 words, as used by the BPE tokenizer.
- _emb_dim_ represents the embedding size, transforming each token into a 768-dimensional vector.
- _n_heads_ indicates the count of attention heads in the multi-head attention mechanism.
- _n_layers_ specifies the number of transformer blocks in the model, which we will cover in the upcoming discussion.
- _drop_rate_ indicates the intensity of the dropout mechanism (0.1 implies a 10% random drop out of hidden units) to prevent overfitting
- _qkv_bias_ determines whether to include a bias vector in the Linear layers of the multi-head attention for query, key, and value computations.

![[Pasted image 20250305102153.png]]

Create the class `DummyGPTModel`

```python
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)


    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
```

Using `tiktoken` we create a batch with 2 texts encodings.

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
```

```shell
tensor([[6109, 3626, 6100, 345],
		[6109, 1110, 6622, 257]])
```

We use the `DummyGPTModel` and feed them with our batch

```python
torch.manual_seed(123)

model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)
```

```shell
Output shape: torch.Size([2, 4, 50257])
tensor([[[-0.9289, 0.2748, -0.7557, ..., -1.6070, 0.2702, -0.5888],
		 [-0.4476, 0.1726, 0.5354, ..., -0.3932, 1.5285, 0.8557],
		 [ 0.5680, 1.6053, -0.2155, ..., 1.1624, 0.1380, 0.7425],
		 [ 0.0447, 2.4787, -0.8843, ..., 1.3219, -0.0864, -0.5856]],

		[[-1.5474, -0.0542, -1.0571, ..., -1.8061, -0.4494, -0.6747],
		 [-0.8422, 0.8243, -0.1098, ..., -0.1434, 0.2079, 1.2046],
		 [ 0.1355, 1.1858, -0.1453, ..., 0.0869, -0.1590, 0.1552],
		 [ 0.1666, -0.8138, 0.2307, ..., 2.5035, -0.3055, -0.3083]]],
		 grad_fn=<UnsafeViewBackward0>)
```

The output tensor has two rows corresponding to the two text samples. Each text sample consists of four tokens; each token is a 50,257-dimensional vector, which matches the size of the tokenizer’s vocabulary.

> The embedding has 50,257 dimensions because each of these dimensions refers to a unique token in the vocabulary. In the post processing code will convert 50,257-dimensional vectors back into token IDs

## Normalizing activations with layer normalization

Implement layer normalization to improve the stability and efficiency of neural network training.

> The main idea behind layer normalization is to adjust the activations (outputs) of a neural network layer to have a mean of 0 and a variance of 1

![[Pasted image 20250305102205.png]]

```python
torch.manual_seed(123)
# Creates two training examples with five dimensions (features) each
batch_example = torch.randn(2, 5)
print(batch_example)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
```

First row lists the layer outputs for the first input and the second row lists the layer outputs for the second row:

```shell
tensor([[-0.1115, 0.1204, -0.3696, -0.2404, -1.1969],
		[ 0.2093, -0.9724, -0.7550, 0.3239, -0.1085]])

tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
		[0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
		grad_fn=<ReluBackward0>)
```

Apply layer normalization to these outputs, let’s examine the mean and variance:

```python
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var
```

```shell
Mean: tensor([[0.1324],
			  [0.2170]], grad_fn=<MeanBackward1>)

Variance: tensor([[0.0231],
				  [0.0398]], grad_fn=<VarBackward0>)
```

> The first row in the mean tensor here contains the mean value for the first input row, and the second output row contains the mean for the second input row.
> ![[Pasted image 20250305102415.png]]

Apply layer normalization to the layer outputs

```python
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)

print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)
```

The normalized layer outputs, which now also contain negative values, have 0 mean and a variance of 1:

```shell
Normalized layer outputs:
 tensor([[-0.8498, -0.8474, -0.8543, -0.8499, -0.8543, -0.8543],
		 [-1.0313, -1.0302, -1.0405, -1.0180, -1.0263, -1.0405]],
		 grad_fn=<DivBackward0>)

Mean:
 tensor([[ 0.0000],
		[ 0.0000]], grad_fn=<MeanBackward1>)

Variance:
 tensor([[1.0000],
		 [1.0000]], grad_fn=<VarBackward0>)
```

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))


    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

This specific implementation of layer normalization operates on the last dimension of the input tensor x, which represents the embedding dimension (`emb_dim`). The variable `eps` is a small constant (`epsilon`) added to the variance to prevent division by zero during normalization. The _scale_ and _shift_ are two trainable parameters (of the same dimension as the input) that the LLM automatically adjusts during training if it is determined that doing so would improve the model’s performance on its training task. This allows the model to learn appropriate scaling and shifting that best suit the data it is processing.

Try the `LayerNorm` module in practice and apply it to the batch input:

```python
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)
```

```shell
Mean:
 tensor([[ -0.0000],
		 [ 0.0000]], grad_fn=<MeanBackward1>)

Variance:
 tensor([[0.8000],
		 [0.8000]], grad_fn=<VarBackward0>)
```

## Implementing a feed forward network with GELU activations

An `activation function` is a mathematical function applied to the output of a neuron. It introduces non-linearity into the model, allowing the network to learn and represent complex patterns in the data.

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
       
 class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )



    def forward(self, x):
        return self.layers(x)


ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)
```

```shell
torch.Size([2, 3, 768])
```

![[Pasted image 20250305102454.png]]

The `FeedForward` module plays a crucial role in enhancing the model’s ability to learn from and generalize the data. Although the input and output dimensions of this module are the same, it internally expands the embedding dimension into a `higherdimensional` space through the first linear layer. This expansion is followed by a nonlinear GELU activation and then a contraction back to the original dimension with the second linear transformation.

## Adding shortcut connections

Shortcut connections, commonly known as **skip connections**, are a crucial concept in deep learning, particularly in deep neural networks like **ResNets (Residual Networks)**. They address problems like the vanishing gradient and degradation, which occur as the depth of a neural network increases.

#### What Are Shortcut Connections?

A shortcut connection skips one or more layers in a neural network and feeds the output of one layer directly into a later layer. These connections bypass intermediate layers and add the input directly to the output of the skipped layers.

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
```

We create a 2 examples using a shortcut and other that didn't:

Without shortcut

```python
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.0]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
print_gradients(model_without_shortcut, sample_input)
```

```shell
layers.0.0.weight has gradient mean of 0.00020173590746708214
layers.1.0.weight has gradient mean of 0.0001201116101583466
layers.2.0.weight has gradient mean of 0.0007152042235247791
layers.3.0.weight has gradient mean of 0.0013988739810883999
layers.4.0.weight has gradient mean of 0.00504964729771018
```

With shortchut

```python
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
```

```shell
layers.0.0.weight has gradient mean of 0.22169791162014008
layers.1.0.weight has gradient mean of 0.20694102346897125
layers.2.0.weight has gradient mean of 0.32896995544433594
layers.3.0.weight has gradient mean of 0.2665732204914093
layers.4.0.weight has gradient mean of 1.3258541822433472
```

As we see when not use de shortcut the gradient turn into more smaller from `layers.4` to `layers.1` , which is a phenomenon called the _vanishing gradient problem_. But if use shortcut we can see that the gradient is more larger from `layers.4` to `layers.1`.

## Connecting attention and linear layers in a transformer block

The idea is that the self-attention mechanism in the multi-head attention block identifies and analyzes relationships between elements in the input sequence. In contrast, the feed forward network modifies the data individually at each position. This combination not only enables a more nuanced understanding and processing of the input but also enhances the model’s overall capacity for handling complex data patterns.

Using the `MultiHeadAttention` make it in the previous chapter we create de `TransformerBlock` class

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):
        # Shortcut connection
        # for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        # Add the original input back
        x = x + shortcut

        # Shortcut connection
        # for feed forward block
        x = shortcut
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        # Add the original input back
        x = x + shortcut
       
        return x
```

## Coding the GPT model

The GPT model architecture implementation

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.post_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

The `__init__` constructor initializes the tokens embeddings and the positional embeddings, the applies the dropout. Next the `__init__` method creates a sequential stack of `TransformerBlock` modules equal to the number of layers `n_layers`. Following the transformer blocks, a `LayerNorm` layer is applied, standardizing the outputs from the transformer blocks to stabilize the learning process.
Finally, a linear output head without bias is defined, which projects the transformer’s output into the vocabulary space of the tokenizer to generate logits for each token in the vocabulary.

The forward method takes a batch of input token indices, computes their embeddings, applies the positional embeddings, passes the sequence through the transformer blocks, normalizes the final output, and then computes the logits, representing the next token’s unnormalized probabilities

#### Initialize the 124-million-parameter GPT model using the GPT_CONFIG_124M

```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
out = model(batch)

print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
```

```shell
Input batch:
 tensor([[6109, 3626, 6100, 345],
		 [6109, 1110, 6622, 257]])

Output shape: torch.Size([2, 4, 50257])
tensor([[[-0.4484, -0.0753, -0.2635, ..., -0.0500, -0.5062, -0.1176],
		 [ 0.5473, -0.5821, -0.6807, ..., -0.5743, -0.5166, -0.0566],
		 [ 0.7189, 0.2278, -0.3215, ..., 0.1400, -0.8558, -0.0778],
		 [-0.8253, 0.4900, -0.3526, ..., 1.1276, 0.1170, -0.2855]],

		[[-0.0954, -0.1975, -0.1253, ..., -0.2928, -0.2195, -0.1345],
		 [ 0.1885, -0.0176, 0.2221, ..., 0.7951, -0.3843, 0.3650],
		 [ 0.7261, 0.6500, -0.2036, ..., 0.4319, 0.0807, -0.1093],
		 [-0.2819, 0.0748, 0.3015, ..., 1.1213, -0.5352, -0.0223]]],
		grad_fn=<UnsafeViewBackward0>)
```

> tensor([[6109, 3626, 6100, 345], <-- Token IDs of text 1

    	 [6109, 1110, 6622, 257]])  <-- Token IDs of text 2

> The output tensor has the shape [2, 4, 50257]

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
```

```shell
Total number of parameters: 163,009,536
```

Here we can see a difference, what de parameters are `163,009,536` and not are `124-million parameters`

**Weight tying** this is a concept that used for GPT-2 and is a architecture that reuse the weights from the token embedding layer in its output layer.

Let's see, the token embeddings and the output layers have the same shape.

```python
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
```

```shell
Token embedding layer shape: torch.Size([50257, 768])
Output layer shape: torch.Size([50257, 768])
```

Remove the output layer parameter count

```python
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())

print(
    f"Number of trainable parameters "
    f"considering weight tying: {total_params_gpt2:,}"
)
```

```shell
Number of trainable parameters considering weight tying: 124,412,160
```

If you have a tensor with a specific shape, such as `[2, 3]`, its size is \( 2 X 3 = 6 \), and that's what `numel()` returns.

```python
import torch
# Create a tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Number of elements in the tensor
print(tensor.numel())  # Output: 6
```

Compute the memory requirements of the 163 million parameters in our `GPTModel` object:

```python
# Calculates the total size in
# bytes (assuming float32, 4
# bytes per parameter)
total_size_bytes = total_params * 4

# Converts to
# megabytes
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")
```

```shell
Total size of the model: 621.83 MB
```

## Generating Text
