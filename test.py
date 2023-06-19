import os
import torchaudio
import librosa
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_audio_files(path, label):
    mfcc_features = []
    labels = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        waveform, sample_rate = torchaudio.load(filepath)
        waveform = waveform.squeeze()

        # Compute MFCC features
        mfcc = librosa.feature.mfcc(y=waveform.numpy(), sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc_features.append(mfcc.T)
        labels.append(label)

    return mfcc_features, labels

def add_context_features(features, context_frames):
    num_examples, max_length, feature_dim = features.shape
    padded_features = np.pad(features, ((0, 0), (context_frames, context_frames), (0, 0)), mode='edge')
    contextual_features = np.zeros((num_examples, max_length, (2 * context_frames + 1) * feature_dim))

    for i in range(num_examples):
        for j in range(max_length):
            start = j
            end = start + 2 * context_frames + 1
            contextual_features[i, j] = padded_features[i, start:end].reshape(-1)

    return contextual_features

#Data paths
sk_path = 'D:\Files\sk'
cs_path = 'D:\Files\cs'
pl_path = 'D:\Files\pl'
be_path = 'D:\Files\s_be'
bg_path = 'D:\Files\s_bg'
hu_path = 'D:\Files\hu'
ru_path = 'D:\Files\s_ru'
sl_path = 'D:\Files\sl'
uk_path = 'D:\Files\s_uk'
sr_path = 'D:\Files\sr'



sk_features, sk_labels = load_audio_files(sk_path, 'sk')
cs_features, cs_labels = load_audio_files(cs_path, 'cs')
# pl_features, pl_labels = load_audio_files(cs_path, 'pl')
# be_features, be_labels = load_audio_files(cs_path, 'be')
# bg_features, bg_labels = load_audio_files(cs_path, 'bg')
# hu_features, hu_labels = load_audio_files(cs_path, 'hu')
# ru_features, ru_labels = load_audio_files(cs_path, 'ru')
# sl_features, sl_labels = load_audio_files(cs_path, 'sl')
# uk_features, uk_labels = load_audio_files(cs_path, 'uk')
# sr_features, sr_labels = load_audio_files(cs_path, 'sr')
#
# features = sk_features + cs_features + pl_features + be_features + bg_features + hu_features + ru_features + sl_features + uk_features + sr_features
# labels = sk_labels + cs_labels + pl_labels + be_labels + bg_labels + hu_labels + ru_labels + sl_labels + uk_labels + sr_labels

features = sk_features + cs_features
labels = sk_labels + cs_labels

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Choose the number of context frames
context_frames = 1

# Pad the MFCC features to a consistent length
max_length = max([len(feature) for feature in features])
# input_size = max_length * 13
input_size = (2 * context_frames + 1) * 13 * max_length


features_padded = np.zeros((len(features), max_length, 13))
for i, feature in enumerate(features):
    features_padded[i, :len(feature), :] = feature




# Add context to your features
contextual_features = add_context_features(features_padded, context_frames)

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features_padded, labels, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(contextual_features, labels, test_size=0.2, random_state=42)


# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 13)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, 13)).reshape(X_test.shape)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader objects
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, dropout_rate=0.3):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        # self.relu3 = nn.ReLU()
        # self.dropout3 = nn.Dropout(dropout_rate)
        # self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        # x = self.relu3(x)
        # x = self.dropout3(x)
        # x = self.fc4(x)
        return x

# input_size = max_length * 13
input_size = (2 * context_frames + 1) * 13 * max_length
hidden_size1 = 64
hidden_size2 = 128  # New hidden layer with 128 units
hidden_size3 = 256
output_size = len(le.classes_)

model = FeedForwardNN(input_size, hidden_size1,hidden_size2, hidden_size3, output_size, dropout_rate=0.3)
criterion = nn.CrossEntropyLoss()

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)

train_losses = []
test_losses = []

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.view(inputs.size(0), -1)



        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct / total

    # Evaluate the model on the test dataset
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.view(inputs.size(0), -1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)
    test_accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# torch.save(model.state_dict(), 'my_model.pth')
print("Training complete.")
# After the training loop
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Losses')
plt.legend()
plt.show()

model.eval()
y_pred = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        y_pred.extend(predicted.tolist())

# Compute the confusion matrix
cm = confusion_matrix(y_test.tolist(), y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# from sklearn.metrics import f1_score


# def predict(model, loader):
#     predicted_labels = []
#     true_labels = []
#
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs = inputs.view(inputs.size(0), -1)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#
#             predicted_labels.extend(predicted.tolist())
#             true_labels.extend(labels.tolist())
#
#     return true_labels, predicted_labels




##################################################################################################
# Load the trained model
# model_path = 'my_model.pth'
# model = FeedForwardNN(input_size, hidden_size, output_size)
# model.load_state_dict(torch.load(model_path))
# model.eval()
#
# # Calculate the score
# correct_predictions = 0
# total_predictions = 0
#
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs = inputs.view(inputs.size(0), -1)
#         outputs = model(inputs)
#         _, predicted = outputs.max(1)
#         correct_predictions += predicted.eq(labels).sum().item()
#         total_predictions += labels.size(0)
#
# score = correct_predictions / total_predictions
# print(f"Score: {score:.4f} ({correct_predictions}/{total_predictions})")
#
# # Create a function to map numeric labels back to language names
# def decode_label(label):
#     return le.inverse_transform([label])[0]
#
#
# # Evaluate the model on each test recording and show the predicted language
# for i, (inputs, labels) in enumerate(test_loader):
#     inputs = inputs.view(inputs.size(0), -1)
#
#     with torch.no_grad():
#         outputs = model(inputs)
#         _, predicted = outputs.max(1)
#
#     for j in range(len(predicted)):
#         print(f"Test recording {i * len(predicted) + j + 1}: "
#               f"True language = {decode_label(labels[j].item())}, "
#               f"Predicted language = {decode_label(predicted[j].item())}")