#!/usr/bin/env python
# coding: utf-8

# In[64]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


# Define dataset paths
train_path = '../Dataset/Train'
test_path = '../Dataset/Test'
val_path = '../Dataset/Validation'


# In[32]:


# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


# In[52]:


# Load datasets using ImageFolder
train_dataset = ImageFolder(root=train_path, transform=data_transform)
test_dataset = ImageFolder(root=test_path, transform=data_transform)
val_dataset = ImageFolder(root=val_path, transform=data_transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


# In[34]:


# Model architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64 * 3, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# In[59]:


# Instantiate the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Move the model to a device (e.g., GPU) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print the model summary
summary(model, input_size=(3, 64, 64))  # Adjust input_size based on your image dimensions


# In[60]:


#Function for generating f1 score
def net_f1score(classifier, test_loader):
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = classifier(images)
            predicted = (outputs > 0.5).float()

            true_labels.extend(labels.float().view(-1, 1).cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    true_labels = np.array(true_labels).flatten()
    predicted_labels = np.array(predicted_labels).flatten()

    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return precision, recall, roc_auc, f1


# In[65]:


# Training the model
num_epochs = 15
for epoch in range(num_epochs):
    print(f"Starting epoch: ",epoch)

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    model.train()

    for entry, data in enumerate(train_loader, 0):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().view(-1, 1))   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training accuracy
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == labels.float().view(-1, 1)).sum().item()
        total_train += labels.size(0)
        
        if entry % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {entry+1}, Loss: {running_loss / 100:.3f}, Training Accuracy: {(correct_train / total_train) * 100:.2f}%')
            running_loss = 0.0

    # Model evaluation on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
    # Perform inference or evaluation without tracking gradients
        for inputs, labels in val_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.float().view(-1, 1)).sum().item()

    accuracy = (correct / total) * 100
    print(f'Validation accuracy: {accuracy:.2f}%')

    print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy:.2f}%')
        
    precision, recall, roc_auc, f1 = net_f1score(model, val_loader)
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'ROC-AUC: {roc_auc:.2f}')
    print(f'F1 Score: {f1:.2f}')
        
    print("-------------------------------------------------------------------------")

print('Finished Training')

# Save the entire model (architecture and weights)
torch.save(model, 'model_ann.pth')


# In[66]:


# Model evaluation on test set
correct = 0
total = 0

# Lists to store true labels and model predictions
true_labels = []
model_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.float().view(-1, 1)).sum().item()
        true_labels.extend(labels.cpu().numpy())
        model_predictions.extend(outputs.cpu().numpy())

accuracy = (correct / total) * 100
print(f'Test accuracy: {accuracy:.2f}%')

