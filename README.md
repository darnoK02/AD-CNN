# SimpleNeuralNetwork.jl

A lightweight neural network library implemented in Julia, featuring automatic differentiation and support for various layer types including Dense, Conv1D, MaxPool1D, and Embedding layers.

## Features

- Custom reverse-mode automatic differentiation
- Support for various layer types:
  - Dense (fully connected)
  - Conv1D (1D convolution)
  - MaxPool1D
  - Embedding
  - Flatten
- Optimizers:
  - SGD (Stochastic Gradient Descent)
  - Adam
- Activation functions:
  - ReLU
  - Sigmoid
  - Tanh
- Loss functions:
  - MSE (Mean Squared Error)
  - Binary Cross Entropy
- Memory-efficient training with gradient clipping
- Batch processing support

## Usage Example

Here's a simple example of creating and training a CNN for text classification:

```julia
using SimpleNeuralNetwork
SNN = SimpleNeuralNetwork

# Create model
model = SNN.Chain(
    SNN.Embedding(vocab_size, embedding_dim),
    SNN.Conv1D(3, embedding_dim, 8, SNN.relu),
    SNN.MaxPool1D(8),
    SNN.Flatten(),
    SNN.Dense(128, 1, SNN.sigmoid)
)

# Initialize optimizer
optimizer = SNN.Adam(learning_rate=0.0005)
SNN.setup(optimizer, model)

# Create data loaders
dataloader_train = SNN.DataLoader((X_train, y_train), batch_size=64, shuffle=true)
dataloader_test = SNN.DataLoader((X_test, y_test), batch_size=64, shuffle=false)

# Train the model
train_losses, test_losses, train_accuracies, test_accuracies = SNN.train!(
    model,
    dataloader_train,
    epochs=5,
    optimizer=optimizer,
    verbose=true,
    test_data=dataloader_test,
    loss_function=:bce
)
```

## Example Results

The model was tested on the IMDB dataset for sentiment analysis. Here are the results from two training runs:

### Run 1
```
Starting training...
Epoch 1: Train Loss: 0.662663 Train Acc: 0.7442 Test Loss: 0.557813 Test Acc: 0.7234 (Epoch time: 33.19s)
Epoch 2: Train Loss: 0.473758 Train Acc: 0.8226 Test Loss: 0.433123 Test Acc: 0.8104 (Epoch time: 22.44s)
Epoch 3: Train Loss: 0.380256 Train Acc: 0.8672 Test Loss: 0.374309 Test Acc: 0.8368 (Epoch time: 22.41s)
Epoch 4: Train Loss: 0.320837 Train Acc: 0.8814 Test Loss: 0.356663 Test Acc: 0.8454 (Epoch time: 22.43s)
Epoch 5: Train Loss: 0.27696 Train Acc: 0.9088 Test Loss: 0.336159 Test Acc: 0.8618 (Epoch time: 22.6s)
Training finished.
```

### Run 2
```
Starting training...
Epoch 1: Train Loss: 0.674532 Train Acc: 0.7328 Test Loss: 0.562773 Test Acc: 0.7206 (Epoch time: 34.21s)
Epoch 2: Train Loss: 0.47439 Train Acc: 0.8282 Test Loss: 0.4249 Test Acc: 0.8036 (Epoch time: 22.43s)
Epoch 3: Train Loss: 0.370657 Train Acc: 0.8662 Test Loss: 0.375468 Test Acc: 0.8336 (Epoch time: 22.79s)
Epoch 4: Train Loss: 0.309282 Train Acc: 0.8876 Test Loss: 0.335519 Test Acc: 0.8544 (Epoch time: 23.07s)
Training finished.
```

The results show consistent improvement in both training and test accuracy, with the model achieving over 85% accuracy on the test set. The training is also relatively fast, with each epoch taking around 22-34 seconds depending on the hardware. 