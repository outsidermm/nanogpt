import torch
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