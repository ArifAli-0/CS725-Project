'''Importing necessary libraries and modules for PyTorch, 
data transformations, datasets, data loaders, image processing,
 evaluation metrics, NumPy, model summary, and plotting.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt

# Part 1 - Building the CNN

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)       #First convolution layer
        self.pool = nn.MaxPool2d(2, 2)    #Max-Pooling layer ( translation invariance)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)   #Second convolution layer
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)   #Third convolution layer
        
        #Adding 2 fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        #Applies convolution layer, applies ReLU activation function to output and then applies max pooling to output of this
        x = self.pool(F.relu(self.conv1(x)))        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
       
        #Reshapes the x using view to flatten it. The -1 is used to infer the size along that dimension, and 32 * 8 * 8 corresponds to the expected size after flattening.
        x = x.view(-1, 32 * 8 * 8)
        
        #Applies the ReLU activation to the result of the fully connected layer
        x = F.relu(self.fc1(x))
        
        #Applies the sigmoid activation function to the result of the second fully connected layer
        x = torch.sigmoid(self.fc2(x))
        return x

#Defining a function to calculate precision, recall, ROC-AUC, and F1 score. This function takes a classifier and a data loader as input.
def net_f1score(classifier, data_loader):
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in data_loader:
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

# Initialize the CNN model
classifier = CNN()

#Printing a summary of the model architecture
summary(classifier, input_size=(3, 64, 64))

# Defining the binary cross-entropy loss and the Adam optimizer for training the model.
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters())

# Part 2 - Fitting the CNN to the images

# Define data transforms
#Resizes the input image to the specified size of (64, 64) pixels.
#Normalizes the tensor values. It subtracts the mean and divides by the standard deviation for each channel.
#(0.5, 0.5, 0.5), (0.5, 0.5, 0.5) represents mean and std. dev of 3 channels (R,G,B).
transform = transforms.Compose([transforms.Resize((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create data loaders for Train, Validation and Test datasets.
train_dataset = ImageFolder('Dataset/Train/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = ImageFolder('Dataset/Validation/', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

test_dataset = ImageFolder('Dataset/Test/', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#initializing list for storing results and we can use that data for analysis purpose.
epochs = []

train_acc = []
train_loss = []
train_f1 = []
train_precision = []
train_recall = [] 
train_roc_auc =[]

valid_acc = []
valid_los = []
valid_f1 = []
valid_precision = []
valid_recall = [] 
valid_roc_auc =[]

print("----------------------------------------------------------------------------")
print('----------------------------Training Started--------------------------------')
print("----------------------------------------------------------------------------")
# Training loop
for epoch in range(20):
    
    epochs.append(epoch+1)
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # Seting model (classifier) to training mode, which enables dropout and batch normalization layers.
    classifier.train()
    
    # 'i' is batch index and data contains input imgs and corresponding labels
    for i, data in enumerate(train_loader, 0):         
        inputs, labels = data
        optimizer.zero_grad()  #clears gradients from previous itration
        outputs = classifier(inputs)
        loss = criterion(outputs, labels.float().view(-1, 1))  #BCE loss between the predictions and actual labels.
        loss.backward()    #Computes the gradients of the loss w.r.t the model parameters.
        optimizer.step()   #Updating the model parameters using the optimizer
        running_loss += loss.item()     #Accumulating the running loss.
        
        # Compute training accuracy
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == labels.float().view(-1, 1)).sum().item()
        total_train += labels.size(0)
            
    # Print statistics and collecting necesssary data in lists after each epoch
    print(f'Train - Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader):.3f}, Training Accuracy: {(correct_train / total_train) * 100:.2f}%')
    
    training_loss = running_loss/len(train_loader)
    train_loss.append(round(float(training_loss),2))
    
    training_accuracy = (correct_train / total_train) * 100
    train_acc.append(round(float(training_accuracy),2))
    
    precision_train, recall_train, roc_auc_train, f1_train = net_f1score(classifier, train_loader)
    
    print(f'Training Precision: {precision_train:.2f}')
    train_precision.append(round(float(precision_train),2))
    
    print(f'Training Recall: {recall_train:.2f}')
    train_recall.append(round(float(recall_train),2))
    
    print(f'Training ROC-AUC: {roc_auc_train:.2f}')
    train_roc_auc.append(round(float(roc_auc_train),2))
    
    print(f'Training F1 Score: {f1_train:.2f}')
    train_f1.append(round(float(f1_train),2))
    
    # Validation phase
    valid_loss = 0.0
    correct_valid = 0
    total_valid = 0
    with torch.no_grad():
        #Setting model to evaluation mode
        classifier.eval()
        for data in valid_loader:
            images, labels = data
            outputs = classifier(images)
            loss = criterion(outputs, labels.float().view(-1, 1))
            valid_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_valid += (predicted == labels.float().view(-1, 1)).sum().item()
            total_valid += labels.size(0)

    avg_valid_loss = valid_loss / len(valid_loader)
    print(f'Validation - Epoch {epoch+1}, Validation Loss: {avg_valid_loss:.3f}, Validation Accuracy: {(correct_valid / total_valid) * 100:.2f}%')
    valid_los.append(round(float(avg_valid_loss),2))
    
    valid_accuracy = (correct_valid / total_valid) * 100
    valid_acc.append(round(float(valid_accuracy),2))
    
    precision_valid, recall_valid, roc_auc_valid, f1_valid = net_f1score(classifier, valid_loader)
    
    print(f'Validation Precision: {precision_valid:.2f}')    
    valid_precision.append(round(float(precision_valid),2))
    
    print(f'Validation Recall: {recall_valid:.2f}')
    valid_recall.append(round(float(recall_valid),2))
    
    print(f'Validation ROC-AUC: {roc_auc_valid:.2f}')
    valid_roc_auc.append(round(float(roc_auc_valid),2))
    
    print(f'Validation F1 Score: {f1_valid:.2f}')
    valid_f1.append(round(float(f1_valid),2))
    
    # Test phase 
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = classifier(images)
            predicted = (outputs > 0.5).float()
            correct_test += (predicted == labels.float().view(-1, 1)).sum().item()
            total_test += labels.size(0)

    print(f'Test - Epoch {epoch+1}, Test Accuracy: {(correct_test / total_test) * 100:.2f}%')
    
    print("----------------------------------------------------------------------------")
    
print('----------------------------Training Finished--------------------------------')
print("----------------------------------------------------------------------------")

# Save the entire model (architecture and weights)
torch.save(classifier, 'model.pth')

#Printing collected data
print(f'Epochs = {epochs}')

print(f'Train acc = {train_acc}')
print(f'Train loss = {train_loss}')
print(f'Train f1 = {train_f1}')
print(f'Train precision = {train_precision}')
print(f'Train recall = {train_recall}') 
print(f'Train ROC-AUC = {train_roc_auc}')

print(f'Valid acc = {valid_acc}')
print(f'Valid Loss = {valid_los}')
print(f'Valid F1 = {valid_f1}')
print(f'Valid Precision = {valid_precision}')
print(f'Valid Recall = {valid_recall}') 
print(f'Valid ROC-AUC = {valid_roc_auc}')

#--------------------------------------------------------------------------------------------------

#Plotting Graphs

#(1) Plotting Training and Validation Accuracy over epochs
plt.figure(figsize=(12, 8))
plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
plt.plot(epochs, valid_acc, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.ylim(bottom = 50, top = 100)
plt.grid(True)
plt.xticks(epochs)
plt.savefig('accuracy.png')
#plt.show()

#(2) Plotting Training and Validation Loss over epochs
plt.figure(figsize=(12, 8))
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, valid_los, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.ylim(bottom = 0, top = 1.00)
plt.grid(True)
plt.xticks(epochs)
plt.savefig('loss.png')
#plt.show()

#(3) Plotting Training Precision and Recall over epochs
plt.figure(figsize=(12, 8))
plt.plot(epochs, train_precision, label='Precision', marker='o')
plt.plot(epochs, train_recall, label='Recall', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Training Precision and Recall over Epochs')
plt.legend()
plt.ylim(bottom = 0.50, top = 1.00)
plt.grid(True)
plt.xticks(epochs)
plt.savefig('train_precision_recall.png')

#(4) Plotting Validation Precision and Recall over epochs
plt.figure(figsize=(12, 8))
plt.plot(epochs, valid_precision, label='Precision', marker='o')
plt.plot(epochs, valid_recall, label='Recall', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Validation Precision and Recall over Epochs')
plt.legend()
plt.ylim(bottom = 0.50, top = 1.00)
plt.grid(True)
plt.xticks(epochs)
plt.savefig('valid_precision_recall.png.png')

#(5) Plotting Training and Validation F1 scores over eopchs
plt.figure(figsize=(12, 8))
plt.plot(epochs, train_f1, label='Training F1', marker='o')
plt.plot(epochs, valid_f1, label='Validation F1', marker='o')
plt.xlabel('Epochs')
plt.ylabel('F1 Scores')
plt.title('Training and Validation F1 scores over Epochs')
plt.legend()
plt.ylim(bottom = 0.50, top = 1.00)
plt.grid(True)
plt.xticks(epochs)
plt.savefig('f1_scores.png')

#(6) Plotting Training ROC-AUC over epochs
plt.figure(figsize=(12, 8))
plt.plot(epochs, train_roc_auc, label='Training ROC-AUC', marker='o')
plt.xlabel('Epochs')
plt.ylabel('ROC-AUC')
plt.title('Training ROC-AUC over Epochs')
plt.legend()
plt.ylim(bottom = 0.50, top = 1.00)
plt.grid(True)
plt.xticks(epochs)
plt.savefig('train_roc_auc.png')

#(7) Plotting Validation ROC-AUC over epochs
plt.figure(figsize=(12, 8))
plt.plot(epochs, valid_roc_auc, label='Validation ROC-AUC', marker='o')
plt.xlabel('Epochs')
plt.ylabel('ROC-AUC')
plt.title('Validation ROC-AUC over Epochs')
plt.legend()
plt.ylim(bottom = 0.50, top = 1.00)
plt.grid(True)
plt.xticks(epochs)
plt.savefig('valid_roc_auc.png')

