https://www.deep-ml.com/problems
## Implement Self-Attention Mechanism
[question](https://www.deep-ml.com/problems/53?from=Deep%20Learning)
[mycode](https://colab.research.google.com/drive/14-kHMSNcP67zwlvQa0OZYxqWTzgFoLEi?usp=sharing)
$$\text{Attention} = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$
$n$: sequence length
$d$: embedding dimension

Complexity: $O(n^2 d)$
- $QK^\top$: $O(n^2 d)$ matrix multiplication of $n*d$ and $d*n$ matrix
- softmax: $O(n^2)$
- times V: $O(n^2 d)$ matrix multiplication of $n*n$ and $n*d$ matrix 

```python
import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
	Q = X @ W_q
	K = X @ W_k
	V = X @ W_v
	return Q, K, V

def self_attention(Q, K, V):
	d_k = Q.shape[1]
	scores = (Q @ K.T) / np.sqrt(d_k)
	scores = scores - scores.max(axis = -1, keepdims = True) 
		# view the scores as scores[i][:] and find the max for each i, i.e., for every row, subtract its own maximum score.
		# keepdims means copying the max n times to keep the dimension
	weights = np.exp(scores) / np.sum(np.exp(scores), axis = -1, keepdims = True)
	attention_output = weights @ V
	return attention_output
	
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
	def __init__(self, embed_dim):
		super().__init__()
		self.embed_dim = embed_dim
		self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
		self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
		self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
		self.scale = embed_dim ** (-0.5)

	def forward(self, x, mask = False):
		b, n, d = x.shape
		Q = self.q_proj(x)
		K = self.k_proj(x)
		V = self.v_proj(x)
		scores = Q @ K.transpose(1, 2) * self.scale 
						# if x: (b, n, d) batch
		weights = F.softmax(scores, dim = -1) # scores.softmax(-1)
		attention_output = weights @ V
		return attention_output
```

The **softmax** function:
$$\text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j}\exp(x_j)}$$
If we have a $n*n$ score matrix $S = [s_1, \ldots, s_n]^\top$ where each row $s_i$ represents the similarity of $Q[i]$ and $K[1,\ldots, n]$, then we want to have softmax function in each row $s_i$, i.e., $$\text{Softmax}(S) = \left[\text{Softmax}(s_1),\ldots, \text{Softmax}(s_n)\right]^\top$$ If we use pytorch we write `F.softmax(scores, dim = -1)` to indicate we are normalizing over the last axis (each column's entries sum to 1).

An improvement for the softmax function: **shift by max** trick $$\text{Softmax}(x_i) = \frac{\exp(x_i - \max(x_j))}{\sum_{j}\exp(x_j - \max(x_j))}$$
这里最大值是取的每一行（所有列）的最大值
- numpy: `score = scores - scores.max(axis = -1, keepdims = True)`
- torch: `scores = scores - scores.max(dim = -1, keepdim = True)[0]`
	- 注意`torch.max`返回的是`values, indices`

注意区别！

|     | np           | torch   |
| --- | ------------ | ------- |
|     | keepdim**s** | keepdim |
|     | axis         | dim     |

### The Pattern Weaver's Code
[question](https://www.deep-ml.com/problems/89?from=Deep%20Learning)
[mycode](https://colab.research.google.com/drive/1751DiKR7dKvZNqCkknLhRhYc5BlQeJJf?usp=sharing)

完全就是attention，重写一遍有几个注意事项
- numpy
	- max是针对每一行
	- 输入如果是list，要`np.array(input).reshape(n,d)` （防止`d=1`）

- torch
	- max返回两项，第一项才是values
		- torch用的是dim和keepdim
	- 如果一开始输入的list是整数，torch.tensor的type自动是`Long(int64)`，为了能和`float32`的matrix做矩阵乘法，我们要自定义type
		- `torch.tensor(x, dtype = float32).view(n,d)`

## Position Encoding
[question](https://www.deep-ml.com/problems/85?from=Deep%20Learning)
[mycode](https://colab.research.google.com/drive/1MfHaBwE4uGX5sCR5NmtNY1G02qoV2E_c?usp=sharing)

假设`pos=3`和`d_model=5`，PE里面的angle是
```python
[[0],
 [1],
 [2]
]
# 和 1/10000^{2*(i//2)/d_model} 相乘，这里i是
[[0],[1],[2],[3],[4]]
```
$$
\begin{align}
\text{PE}(\text{pos}, 2i) = &\sin(\text{pos}/10000^{2i/d_{\text{model}}}) \\
\text{PE}(\text{pos}, 2i+1) = &\cos(\text{pos}/10000^{2i/d_{\text{model}}})
\end{align}
$$
注意这里无论是`2i`还是`2i+1`，用的angle都是`2i`的，所以angle里面要用`2*(i//2)`

```python
import numpy as np

def pos_encoding(position: int, d_model: int):
	if position == 0 or d_model <= 0:
		return -1

	# position: dimension n
	# d_model: dimension d
	pos = np.arange(position).reshape(-1,1) # n*1
	i = np.arange(d_model).reshape(1,-1) # 1*d
	
	# angle_rads = 1 / np.power(10000, (2 * (i // 2)) / d_model) # 1*d
	angles = np.exp(-2*(i//2) / d_model * np.log(10000))
	pos_encoding = pos * angle_rads #n*d
	
	pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
	pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
	pos_encoding = np.float16(pos_encoding)
	return pos_encoding
```

用到的几个技巧：
1. broadcasting `A +-*/ B`
	- 两种情况：每个维度 要么相等，要么其中一个是1
			- 例子：`A.shape=(5,1), B.shape=(1,4), result shape (5,4)`
			- 例子：`A.shape=(5,2), B.shape=(1,2), result shape (5,2)`
	```python
	import numpy as np
	
	A = np.array([[1],
				  [2],
				  [3],
				  [4],
				  [5]])  # shape (5, 1)
	
	B = np.array([[10, 20, 30, 40]])  # shape (1, 4)
	
	result = A + B  # broadcasts to shape (5, 4)
	print(result)
	
	#[[11 21 31 41]
	 #[12 22 32 42]
	 #[13 23 33 43]
	 #[14 24 34 44]
	 #[15 25 35 45]]
	```

第二个技巧是 
- `[0::2]`表示0开始，每隔2个index（偶数位）(0,2,4...)
- `[1::2]`表示1开始，每隔2个index（奇数位）(1,3,5...)

第三个技巧（improvement）是（防止over/ underflow）:
$$b^x = e^{x\log b}$$
- it is better to write $1/10000^{2i/d_{\text{model}}} = e^{-2i\log 10000/d_{\text{model}}}$ 
	`torch.exp(-2*(i//2) * math.log(10000) / d_model)`
		注意`torch.log`的input必须是Tensor，还是用`math.log`吧

因为`angle_rads`都一样，所以也可以写成
```python
i = torch.arange(0, d_model, 2).reshape(1, -1)
angle_rads = torch.exp(-i/d_model * math.log(10000))
```

```python
import torch
import math

def positional_encoding(position, d_model, device = 'cpu'):
	pe = torch.zeros(position, d_model, device = device)
	pos = torch.arange(position).reshape(-1, 1)
	# 也可以写.unsqueeze(1), 1表示新加dimension的位置
	i = torch.arange(0, d_model, 2).reshape(1, -1)
	div_term = torch.exp(-i/d_model * math.log(10000))
	
	pe[:, 0::2] = torch.sin(pos * div_term)
	pe[:, 1::2] = torch.cos(pos * div_term)
	return pe.to(dtype=torch.float16)
```

把positional encoding写成nn.Module
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
	def __init__(self, max_len=5000, d_model, dropout=0.1):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		pos = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
		pe[:, 0::2] = torch.sin(pos * div_term)
		pe[:, 1::2] = torch.cos(pos * div_term)
		pe = pe.unsqueeze(0) # shape: (1, max_len, d_model)
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		# x (Tensor): shape (batch_size, seq_len, d_model)
		x = x + self.pe[:, :x.size(1), :]
		return self.dropout(x)
```

| Feature                       | `register_buffer` | `nn.Parameter` |
| ----------------------------- | ----------------- | -------------- |
| Included in `state_dict()`    | ✅                 | ✅              |
| Moved by `.to()` or `.cuda()` | ✅                 | ✅              |
| Trained by optimizer          | ❌                 | ✅              |

## Multi-Head Attention
[question](https://www.deep-ml.com/problems/94?from=Deep%20Learning)
[mycode](https://colab.research.google.com/drive/1RBgjKGtnqKIhcGfQMDwFB6qCCdBnlueB?usp=sharing)

```python
def multi_head_attention(Q, K, V, n_heads):
	d_model = Q.shape[-1]
	assert d_model % n_heads == 0
	d_k = d_model // n_heads
	
	new_Q = Q.reshape(-1, n_heads, d_k).transpose(1, 0, 2)
	new_K = K.reshape(-1, n_heads, d_k).transpose(1, 0, 2)
	new_V = V.reshape(-1, n_heads, d_k).transpose(1, 0, 2)

	attention = self_attention(new_Q, new_K, new_V)
	return attention.transpose(1,0, 2).reshape(-1, d_model)
```

reshape的时候都是（个数，维度），都是个数在前，维度在后

torch的代码也很相似
```python
def split_heads(x):
	return x.reshape(batch_size, seq_len, n_heads, d_k).transpose(1,2)
	# 返回值的size(): (batch_size, n_heads, seq_len, d_k)
```

## GPT-2 Text Generation
[question](https://www.deep-ml.com/problems/94?from=Deep%20Learning)
[mycode](https://colab.research.google.com/drive/1RBgjKGtnqKIhcGfQMDwFB6qCCdBnlueB?usp=sharing)

```python
def gelu(x):
	return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x ** 3)))

  
def softmax(x):
	shifted_x = x - np.max(x, axis = -1, keepdims = True)
	exp_x = np.exp(shifted_x)
	return exp_x / np.sum(x, axis = -1, keepdims = True)


def layernorm(x, g, b, epsilon = 1e-9):
	mu = np.mean(x, axis = -1, keepdims = True)
	sigma2 = np.var(x, axis = -1, keepdims = True)
	out = g * (x-mu) / np.sqrt(sigma2 + epsilon) + b
	return out

  
def linear(x, w, b):
	return x @ w + b

  
def ffn(x, c_fc, c_proj):
	return linear(gelu(linear, **c_fc), **c_proj)

  
def attention(q, k, v, mask):
	dk = q.shape[-1]
	attn = softmax(q @ k.T / np.sqrt(dk) + mask) @ v

  
def mha(x, c_attn, c_proj, n_head):
	x = linear(x, **c_attn)
	qkv_heads = list(map(lambda a: np.split(a, n_head, axis = -1), np.split(x, 3, axis = -1)))
	mask = (1 - np.tri(x.shape[0], dtype = x.dtype)) * -1e10
	out = [attention(q, k, v, mask) for q, k, v in zip(*qkv_heads)]
	x = linear(np.hstack(out), **c_proj)
	return x

  
def transformer(x, mlp, attn, ln_1, ln_2, n_head):
	x = x + mha(layernorm(x, **ln_1), **attn, n_head = n_head)
	x = x + ffn(layernorm(x, **ln_2), **mlp)
	return x

  
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
	x = wte[inputs] + wpe[range(len(inputs))]
	for block in blocks:
		x = transformer_block(x, **block, n_head = n_head)
	return layernorm(x, **ln_f) @ wte.T

  
def generate(inputs, params, n_head, n_tokens_to_generate):
	for _ in range(n_tokens_to_generate):
		logits = gpt2(inputs, **params, n_head = n_head)
		next_id = np.argmax(logits[-1])
		inputs.append(int(next_id))
	return inputs[len(inputs) - n_tokens_to_generate:]

  
def gen_text(prompt: str, n_tokens_to_generate: int = 40):
	####your code goes here####
	np.random.seed(42)
	encoder, hparams, params = load_encoder_hparams_and_params()
	input_ids = encoder.encode(prompt)
	assert len(input_ids) + n_tokens_to_generate < hparams['n_ctx']
	output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
	output_text = encoder.decode(output_ids)
	return output_text

  
def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
	class DummyBPE:
		def __init__(self):
			self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

		def encode(self, text: str):
			tokens = text.strip().split()
			return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

		def decode(self, token_ids: list):
			reversed_dict = {v: k for k, v in self.encoder_dict.items()}
			return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

  
	hparams = {
		"n_ctx": 1024,
		"n_head": 12
	}
	
	params = {
		"wte": np.random.rand(3, 10),
		"wpe": np.random.rand(1024, 10),
		"blocks": [],
	
		"ln_f": {
		"g": np.ones(10),
		"b": np.zeros(10),
		}
	}
	
	encoder = DummyBPE()
	return encoder, hparams, params
```