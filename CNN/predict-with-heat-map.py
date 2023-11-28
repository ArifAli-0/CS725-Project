from google.colab.patches import cv2_imshow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

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

# Grad-CAM class

class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name

        if self.layer_name is None:
            self.layer_name = self.find_target_layer()

    def find_target_layer(self):
        # Find the final convolutional layer in the network
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                return name

        # If no convolutional layer is found
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        # Forward pass
        image.requires_grad = True
        output = self.model(image)

        if class_idx is None:
            class_idx = torch.argmax(output).item()

        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)

        # Get the gradient of the output with respect to the model's parameters
        grads = image.grad

        # Get the weights of the target layer
        weights = grads.mean(dim=[2, 3], keepdim=True).mean(dim=[0, 1, 2, 3], keepdim=True)

        # Get the activation map of the target layer
        activation = getattr(self.model, self.layer_name)(image)

        # Compute the weighted sum of the activation map
        heatmap = (weights * activation).sum(dim=1, keepdim=True)

        # Apply ReLU to the heatmap
        heatmap = F.relu(heatmap)

        # Normalize the heatmap
        heatmap_max = torch.max(heatmap)
        heatmap_min = torch.min(heatmap)
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + eps)

        # Convert to NumPy array
        heatmap = heatmap.squeeze().detach().numpy()

        return heatmap



# Load the trained model
classifier = torch.load('sample_data/model.pth')
classifier.eval()

# Loop through all images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)

        # Load and preprocess the test image
        test_image = Image.open(image_path)
        input_tensor = transform(test_image).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            prediction = classifier(input_tensor).item()

        # Determine result based on the prediction threshold (0.5)
        result = 'real' if prediction > 0.5 else 'fake'

        # Grad-CAM
        grad_cam = GradCAM(classifier)
        heatmap = grad_cam.compute_heatmap(input_tensor)

        # Resize the heatmap to match the size of the original image
        heatmap = cv2.resize(heatmap, (test_image.width, test_image.height))

        # Convert the heatmap to the 'hot' colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        # Overlay the heatmap on the original image
        output_image = cv2.addWeighted(cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR), 0.5, heatmap_colored, 0.5, 0)

        # Display the results
        print(f"Filename: {filename}, Prediction: {result}")
        cv2_imshow(cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR))
        cv2_imshow(heatmap_colored)
        cv2_imshow(output_image)
        print("-------------------------------------------------------------------------")
