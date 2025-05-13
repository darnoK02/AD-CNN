import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

DATA_FILE = '../data/imdb_dataset_prepared.json'
EMBEDDING_DIM = 128
HIDDEN_DIM_CNN = 128
KERNEL_SIZE = 3
HIDDEN_DIM_DENSE = 64
OUTPUT_DIM = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

print("Loading data...")
try:
    with open(DATA_FILE, 'r') as f:
        data_json = json.load(f)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please ensure 'imdb_dataset_prepared.json' is in the 'data' directory relative to the script location.")
    exit()

X_train = np.array(data_json['X_train'], dtype=np.int64)
y_train = np.array(data_json['y_train'], dtype=np.float32)
X_test = np.array(data_json['X_test'], dtype=np.int64)
y_test = np.array(data_json['y_test'], dtype=np.float32)

try:
    VOCAB_SIZE = data_json['vocab_size']
except KeyError:
    print("'vocab_size' not found in JSON, deriving from data...")
    if X_train.size > 0 and X_test.size > 0:
        VOCAB_SIZE = int(max(np.max(X_train), np.max(X_test))) + 1
    elif X_train.size > 0:
        VOCAB_SIZE = int(np.max(X_train)) + 1
    elif X_test.size > 0:
        VOCAB_SIZE = int(np.max(X_test)) + 1
    else:
        print("Error: X_train and X_test are empty. Cannot derive VOCAB_SIZE.")
        exit()

try:
    MAX_LEN = data_json['maxlen']
except KeyError:
    print("'maxlen' not found in JSON, deriving from X_train shape...")
    if X_train.ndim == 2 and X_train.shape[1] > 0:
        MAX_LEN = X_train.shape[1]
    else:
        print(f"Error: X_train has unexpected shape {X_train.shape}. Cannot derive MAX_LEN.")
        exit()

X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train).unsqueeze(1)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_size, hidden_dim, output_dim, max_len, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=filter_size)
        self.relu = nn.ReLU()
        self.conv_out_len = max_len - filter_size + 1
        if self.conv_out_len <= 0:
            raise ValueError(f"Kernel size {filter_size} is too large for sequence length {max_len}. Results in non-positive conv output length.")
        self.pool = nn.MaxPool1d(kernel_size=self.conv_out_len)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_filters, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_indices):
        embedded = self.embedding(text_indices)
        embedded_permuted = embedded.permute(0, 2, 1)
        conved = self.conv1d(embedded_permuted)
        activated = self.relu(conved)
        pooled = self.pool(activated)
        flat = self.flatten(pooled)
        dense1 = self.relu(self.fc1(flat))
        dropped_out = self.dropout(dense1)
        output = self.fc2(dropped_out)
        return output

model = CNNTextClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM_CNN, KERNEL_SIZE, HIDDEN_DIM_DENSE, OUTPUT_DIM, MAX_LEN).to(DEVICE)
criterion = nn.BCEWithLogitsLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def calculate_accuracy(preds, y):
    predicted_classes = torch.sigmoid(preds) > 0.5
    correct = (predicted_classes == y.bool()).float()
    acc = correct.sum() / len(correct)
    return acc

print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch_idx, (texts, labels) in enumerate(train_loader):
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        acc = calculate_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
            
    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_acc = epoch_acc / len(train_loader)
    model.eval()
    test_loss = 0
    test_acc = 0
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)
            test_loss += loss.item()
            test_acc += acc.item()
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    epoch_time = time.time() - epoch_start_time

    print(f'Epoch {epoch+1}/{NUM_EPOCHS} [{epoch_time:.1f}s]')
    print(f'  Train Loss: {avg_epoch_loss:.4f} | Train Acc: {avg_epoch_acc:.4f}')
    print(f'  Test Loss:  {avg_test_loss:.4f} | Test Acc:  {avg_test_acc:.4f}')
    print('-' * 60)

print("\nTraining finished.")

# Example testing run output:
# Epoch 1/5 [16.7s]
#   Train Loss: 0.5062 | Train Acc: 0.7448
#   Test Loss:  0.3782 | Test Acc:  0.8296
# ------------------------------------------------------------
# Epoch 2/5 [16.5s]
#   Train Loss: 0.3245 | Train Acc: 0.8658
#   Test Loss:  0.3271 | Test Acc:  0.8568
# ------------------------------------------------------------
# Epoch 3/5 [17.8s]
#   Train Loss: 0.2125 | Train Acc: 0.9216
#   Test Loss:  0.3577 | Test Acc:  0.8598
# ------------------------------------------------------------
# Epoch 4/5 [17.5s]
#   Train Loss: 0.1185 | Train Acc: 0.9606
#   Test Loss:  0.4290 | Test Acc:  0.8546
# ------------------------------------------------------------
# Epoch 5/5 [17.6s]
#   Train Loss: 0.0574 | Train Acc: 0.9811
#   Test Loss:  0.4755 | Test Acc:  0.8540
# ------------------------------------------------------------