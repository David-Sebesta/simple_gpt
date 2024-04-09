# Simple_GPT

Simple_GPT is a simple decoder only Generative Pre-Trained Transformer model. This was an exercise to learn more about self-attention models. It uses tiny shakespeare as its dataset, and generates a file in the same speach pattern. There are three main files: tokenizer.py, model.py, and train.py.

Tokenizer.py contains a tokenizer that can use a custom regex expression to customize the way the tokens are split. It then uses a byte pair encoding algorithm to create the tokens based on the most common pairs of characters.

Model.py contains a simple GPT model. It uses a causal multihead attention and a simple feedforward multi-layer perceptron with a gaussian error linear unit as its activation function.

Train.py loads the data, creates the model, and trains it. It then generates a file based on a zero context input into the model.

The model ends up overfitting quite a bit since its training loss is much less than its validation loss. This could be fixed by tuning the hyperparameters and by having a larger data set for both training and validation.


References:  
Attention is All you Need by Ashish Vaswant  
Andrej Karpthay's video https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5402s
