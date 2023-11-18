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

# Part 1 - Building the CNN

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

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

# Initialize the CNN model
classifier = CNN()

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters())

# Part 2 - Fitting the CNN to the images

# Define data transforms
transform = transforms.Compose([transforms.Resize((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create data loaders
train_dataset = ImageFolder('train/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

test_dataset = ImageFolder('test/', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Training loop
for epoch in range(25):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Compute training accuracy
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == labels.float().view(-1, 1)).sum().item()
        total_train += labels.size(0)
        
        if i % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}, Training Accuracy: {(correct_train / total_train) * 100:.2f}%')
            running_loss = 0.0

    # Compute test accuracy
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = classifier(images)
            predicted = (outputs > 0.5).float()
            correct_test += (predicted == labels.float().view(-1, 1)).sum().item()
            total_test += labels.size(0)

    print(f'Epoch {epoch+1}, Test Accuracy: {(correct_test / total_test) * 100:.2f}%')
    
    precision, recall, roc_auc, f1 = net_f1score(classifier, test_loader)
    #print(f'Epoch {epoch+1}, Test Accuracy: {(correct_test / total_test) * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'ROC-AUC: {roc_auc:.2f}')
    print(f'F1 Score: {f1:.2f}')
    
    print("-------------------------------------------------------------------------")
print('Finished Training')

# Save the entire model (architecture and weights)
torch.save(classifier, 'model.pth')

# Part 3 - Making new predictions



# Load and preprocess the test image
test_image = Image.open('predict/real1.jpg')
transform = transforms.Compose([transforms.Resize((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_image = transform(test_image).unsqueeze(0)

# Make a prediction
with torch.no_grad():
    prediction = classifier(test_image).item()

if prediction > 0.5:
    result = 'real'
else:
    result = 'fake'

print(result)

