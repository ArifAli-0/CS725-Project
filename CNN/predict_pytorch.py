import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

#Define the model architecture
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

# Initialize
classifier = CNN()


# Load the entire model (architecture and weights)
classifier = torch.load('model.pth')
classifier.eval()  # Set the model to evaluation mode

# Define data transform for the input image
transform = transforms.Compose([transforms.Resize((128, 128)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

folder_path = 'generated-images'

# Loop through all JPG files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)

        # Open and transform the image
        test_image = Image.open(image_path)
        test_image = transform(test_image).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            prediction = classifier(test_image).item()

        # Determine result based on the prediction threshold (0.5)
        result = 'real' if prediction > 0.5 else 'fake'
        
        # Print or store the result
        print(f"{filename}: {result}")
