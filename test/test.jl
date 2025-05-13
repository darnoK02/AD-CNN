import Pkg
Pkg.activate("..")
Pkg.add("JLD2")
Pkg.add("SpecialFunctions")
using JLD2
using Printf
using Random
Pkg.instantiate()

include("../src/SimpleNeuralNetwork.jl")
import .SimpleNeuralNetwork
SNN = SimpleNeuralNetwork

data_path = joinpath(@__DIR__, "..", "data", "imdb_dataset_prepared.jld2")
X_train = load(data_path, "X_train")
y_train = load(data_path, "y_train")
X_test = load(data_path, "X_test")
y_test = load(data_path, "y_test")
embeddings = load(data_path, "embeddings")
vocab = load(data_path, "vocab")
embedding_dim = size(embeddings, 1)
vocab_size = length(vocab)

model = SNN.Chain(
    SNN.Embedding(vocab_size, embedding_dim),
    SNN.Conv1D(3, embedding_dim, 8, SNN.relu),
    SNN.MaxPool1D(8),
    SNN.Flatten(),
    SNN.Dense(128, 1, SNN.sigmoid)
)

model.layers[1].weights.value .= embeddings


epochs = 4
batch_size = 64
optimizer = SNN.Adam(learning_rate=0.0005)
SNN.setup(optimizer, model)

dataloader_train = SNN.DataLoader((X_train, y_train), batch_size=batch_size, shuffle=true)
dataloader_test = SNN.DataLoader((X_test, y_test), batch_size=batch_size, shuffle=false)

train_losses, test_losses, train_accuracies, test_accuracies = SNN.train!(
    model,
    dataloader_train,
    epochs,
    optimizer=optimizer,
    verbose=true,
    test_data=dataloader_test,
    loss_function=:bce
)

# Example testing run output:
# Starting training...
# Epoch 1: Train Loss: 0.662663 Train Acc: 0.7442 Test Loss: 0.557813 Test Acc: 0.7234 (Epoch time: 33.19s)
# Epoch 2: Train Loss: 0.473758 Train Acc: 0.8226 Test Loss: 0.433123 Test Acc: 0.8104 (Epoch time: 22.44s)
# Epoch 3: Train Loss: 0.380256 Train Acc: 0.8672 Test Loss: 0.374309 Test Acc: 0.8368 (Epoch time: 22.41s)
# Epoch 4: Train Loss: 0.320837 Train Acc: 0.8814 Test Loss: 0.356663 Test Acc: 0.8454 (Epoch time: 22.43s)
# Epoch 5: Train Loss: 0.27696 Train Acc: 0.9088 Test Loss: 0.336159 Test Acc: 0.8618 (Epoch time: 22.6s)
# Training finished.