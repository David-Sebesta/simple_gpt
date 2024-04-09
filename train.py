import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import GPT
from tokenizer import Tokenizer

TRAIN_MODEL = False
MODEL_PATH = "gpt_model_weights.pth"

# Hyperparameters

# Network Tuning parameters
batch_size = 64 # 
block_size = 256 # Context length for predictions
n_embd = 384 # length of embedding dimension
n_head = 6 # number of heads
n_layer = 6 # number of layers
dropout_freq = 0.2
vocab_size = 500

# Training Parameters
lr = 3e-4 # Learning Rate
max_epochs = 5000 #
evaluation_interval = 500 #
evaluation_iterations = 200

torch.manual_seed(4269)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#######
with open('miini_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    

tokenizer = Tokenizer()
tokenizer.train(text, vocab_size=vocab_size)


data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Custom data loading function
def get_batch(split : str) -> tuple[torch.Tensor, torch.Tensor]:
    """ Gets a small batch of data for inputs x and targets y
        x is of length block_size
        
        y is of length block_size and its value follows
        x[n] = y[n+1]

    Args:
        split (str): If split == 'train', it gets training data otherwise
                        it returns validation data

    Returns:
        tuple[torch.Tensor, torch.Tensor]: x_batch, y_batch
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss() -> dict: 
    """Estimate the loss of a range

    Returns:
        dict: A dict with train loss and validation loss
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iterations)
        for k in range(evaluation_iterations):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(vocab_size=vocab_size)
m = model.to(device)

if TRAIN_MODEL:
        
    print(sum(p.numel() for p in m.parameters())/1e6, 'M Parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # fit the model
    for epoch in range(max_epochs):        
        # evaluation
        if epoch % evaluation_interval == 0 or epoch == max_epochs - 1:
            losses = estimate_loss()
            print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
        x_batch, y_batch = get_batch('train')
            
        # evaluate loss
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), MODEL_PATH)

else:
    model.load_state_dict(torch.load(MODEL_PATH))

### Zero context
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(tokenizer.decode(model.generate(context, max_new_tokens=5000)[0].tolist()))


with open("output\\zero_context.txt", "w") as f:
    text = tokenizer.decode(model.generate(context, max_new_tokens=5000)[0].tolist())
    f.write(text)