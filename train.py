import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
print("The length of dataset in characters ", len(text))

print(text[:1000])

chars=sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# TODO: Tiktoken usage
# Tokeniser - Character to integer
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hi there"))
print(decode(encode("hi there")))

# Encode tiny shakespeare ds and wrap in tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype) 
print(data[:1000])

# DS 90/10 val/train split
n = int (0.9*len(text))
train_data = data[:n]
val_data = data[n:]

# Known as block size. i.e. from [1, context_size] amounts of character can the LLM output its prediction of the next token
# Have to group training data in chunks of size context_size + 1 to traing and predict with context_size examples
context_size = 8
train_data[:context_size+1]

# Batching - manually done ._.
batch_size = 4

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)- context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    return x,y

xb, yb = get_batch("train")
print("inputs: ", xb.shape, "targets: ", yb.shape)

for b in range(batch_size):
    for t in range(context_size):
        # Note: Splicing inconclusive of end bound hence t+1 rather than t
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")

# Using one token to predict the possibility of the following token
class BiagramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self,idx,targets = None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Using view here instead of reshape since the tensors are contiguous in memory
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits, loss
    
    def generate(self, idx, max_new_token):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_token):
            logits,loss = self(idx)
            #last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx

model = BiagramLanguageModel(vocab_size=vocab_size)
logits, loss = model(xb,yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1,1), dtype=torch.long)
result = model.generate(idx, max_new_token=100)[0].tolist()
print(decode(result))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
 
batch_size = 32
for steps in range(20000):
    #sample a batch of data
    xb, yb = get_batch("train")
    #evaluate the loss
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(steps, loss.item())
    
idx = torch.zeros((1,1), dtype=torch.long)
result = model.generate(idx, max_new_token=500)[0].tolist()
print(decode(result))