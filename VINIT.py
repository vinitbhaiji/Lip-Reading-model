#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
import dlib
import numpy as np
from scipy import ndimage
import os
import cv2
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split

# Load video
cap = cv2.VideoCapture(0)

# Facial Landmark Extraction using dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Check if video file was successfully loaded
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Set the desired frame rate to 30 fps
cap.set(cv2.CAP_PROP_FPS, 30)

# Get the actual frame rate after setting
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frame rate: {:.2f}".format(fps))

# Set the frame number to start capturing from
frame_number = 0

# Loop through the video frames
while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()

    # Check if frame was successfully read
    if not ret:
        break

    # Increment the frame number
    frame_number += 1

    
    # Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        continue
    
    print(f"Detected {len(faces)} face(s) in frame {frame_number}")
    
    # Face Tracking
    face = faces[0]
    landmarks = predictor(gray, face)
    
    # Facial Landmark Detectionqqqqqqqqqqq
    points = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        points[i] = (landmarks.part(i).x, landmarks.part(i).y)
    
    # Greyscale Conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Scaling
    scale_factor = 0.5
    resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # Central Cropping
    h, w = resized.shape
    crop_size = int(min(h, w) * 0.9)
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    bottom = top + crop_size
    right = left + crop_size
    cropped = resized[top:bottom, left:right]
    
    # Horizontal Flipping
    flip = cv2.flip(cropped, 1)
    
    # Random Frame Removal
    if np.random.rand() > 0.9:
        continue
    
    # Pixel Shifting
    shift_factor = 0.1
    dx, dy = np.random.randint(-shift_factor * w, shift_factor * w, size=2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(flip, M, (w, h))
    
    # z-Score Normalization
    normalized = (shifted - np.mean(shifted)) / np.std(shifted)
    

    # Perform lip reading on the current frame here (optional)

    # Save the current frame as an image file
    image_file = f"frame{frame_number}.jpeg"
    path = 'E:/mlops/Lip_Reading/RESEARCH PAPER_CODE/cap_images'
    cv2.imwrite(os.path.join(path,image_file), frame)
    
    # Display preprocessed image
    cv2.imshow('Preprocessed Image', frame)  # Show the frame in color
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.imshow('Preprocessed Image', normalized) 
    
    # key = cv2.waitKey(1)
    # if key == ord('q'):
    #     break
        


# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Define the spatial-temporal visual frontend for viseme classification
class VisualFrontendViseme(nn.Module):
    def __init__(self,num_channels=3):
        super(VisualFrontendViseme, self).__init__()
        # Add spatial-temporal visual frontend layers for viseme classification here
        self.conv1 = nn.Conv3d(num_channels, 64, kernel_size=(5, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.residual_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.residual_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.residual_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.residual_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.residual_conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.residual_conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.residual_conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.residual_conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, visual_input):
        x = self.conv1(visual_input)
        x = self.pool(x)
        x = self.residual_conv1(x)
        x = self.residual_conv2(x)
        x = self.residual_conv3(x)
        x = self.residual_conv4(x)
        x = self.residual_conv5(x)
        x = self.residual_conv6(x)
        x = self.residual_conv7(x)
        x = self.residual_conv8(x)
        return x

# Define the attention-based transformer for viseme classification
class TransformerViseme(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers):
        super(TransformerViseme, self).__init__()
        # Add your attention-based transformer layers for viseme classification here
        self.self_attn = nn.MultiheadAttention(input_size, num_heads)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)

    def forward(self, viseme_input):
        x, _ = self.self_attn(viseme_input, viseme_input, viseme_input)
        x = self.norm1(viseme_input + self.dropout(x))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.norm2(x + self.dropout(x))
        return x

# Define the viseme classifier
class VisemeClassifier(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, num_classes):
        super(VisemeClassifier, self).__init__()
        self.visual_frontend = VisualFrontendViseme()
        self.transformer = TransformerViseme(input_size, num_heads, hidden_size, num_layers)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, visual_input):
        visual_features = self.visual_frontend(visual_input)
        transformed_visemes = self.transformer(visual_features)
        viseme_output = self.fc(transformed_visemes[:, -1, :])  # Use the last time step as input
        return viseme_output

# Define the attention-based transformer for word detection
class TransformerWord(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, num_classes):
        super(TransformerWord, self).__init__()
        # Add your attention-based transformer layers for word detection here
        self.self_attn = nn.MultiheadAttention(input_size, num_heads)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, viseme_input):
        x = viseme_input.transpose(0, 1)  # Reshape for self-attention
        x, _ = self.self_attn(x, x, x)
        x = x.transpose(0, 1)  # Reshape back to the original shape
        x = self.norm1(viseme_input + self.dropout(x))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.norm2(viseme_input + self.dropout(x))
        word_output = self.fc(x[:, -1, :])  # Use the last time step as input
        return word_output

# Define the overall lip reading system
class LipReadingSystem(nn.Module):
    def __init__(self, viseme_input_size, viseme_num_heads, viseme_hidden_size, viseme_num_layers, viseme_num_classes,
                word_input_size, word_num_heads, word_hidden_size, word_num_layers, word_num_classes):
        super(LipReadingSystem, self).__init__()
        self.viseme_classifier = VisemeClassifier(viseme_input_size, viseme_num_heads, viseme_hidden_size, viseme_num_layers, viseme_num_classes)
        self.word_detector = TransformerWord(word_input_size, word_num_heads, word_hidden_size, word_num_layers, word_num_classes)

    def forward(self, visual_input):
        viseme_output = self.viseme_classifier(visual_input)
        predicted_words = self.word_detector(viseme_output)
        return predicted_words

# Define the dataset and data loaders
class LipReadingDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        visual_input = self.data[index]
        viseme_label = self.labels[index]

        if self.transform:
            visual_input = self.transform(visual_input)

        return visual_input, viseme_label



# Specify the directory path
#data = ["data1.mp4", "data2.mp4", "data3.mp4", "data4.mp4", "data5.mp4"]  # Replace with your data
#labels = [0, 1,2,3,4]  # Replace with your labels



# Set the directory path containing your images
image_directory = 'E:/mlops/Lip_Reading/RESEARCH PAPER_CODE/cap_images/'  # Replace with the path to your image folder

# Set the directory path containing your labeled images
image_directory = 'E:/mlops/Lip_Reading/RESEARCH PAPER_CODE/Input_Image_To_Algo/'  # Replace with the path to your image folder

# Set the desired width and height
new_width = 256
new_height = 256

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Iterate through the image files and process each one
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(image_directory, image_file)

    # Load the image with error handling
    image = cv2.imread(image_path)

    if image is not None:
        # Resize the image to the desired dimensions
        resized_image = cv2.resize(image, (new_width, new_height))

        # Save the resized image in a new directory or overwrite the original
        output_path = os.path.join('E:/mlops/Lip_Reading/RESEARCH PAPER_CODE/Input_Image_To_Algo/', image_file)  # Change 'output_folder' to your desired output folder
        cv2.imwrite(output_path, resized_image)
    else:
        print(f"Skipped {image_file}: Image not found or could not be loaded.")




# Create lists to store the images and labels
image_list = []
labels = []

# Iterate through the image files and load them
for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Extract the label from the filename (assuming labels are before an underscore)
        label = re.split(r'_|\.', filename)[0]

        # Construct the full path to the image
        image_path = os.path.join(image_directory, filename)

        # Load the image with error handling
        image = cv2.imread(image_path)

        if image is not None:
            image_list.append(image)  # Add the loaded image to the list
            labels.append(label)  # Add the corresponding label to the labels list
        else:
            print(f"Skipped {filename}: Image not found or could not be loaded.")

# Now, image_list contains all the loaded images, and labels contains the corresponding labels



# Split the dataset into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(image_list, labels, test_size=0.2, random_state=42)


# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((224, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.RandomCrop((224, 256)),  # Fixed the crop size
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
    # Add more transformations as needed
])

# Create dataset instances
train_dataset = LipReadingDataset(train_data, train_labels, transform=transform)
test_dataset = LipReadingDataset(test_data, test_labels, transform=transform)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Create an instance of the LipReadingSystem
lip_reading_system = LipReadingSystem(viseme_input_size=64, viseme_num_heads=4, viseme_hidden_size=256, viseme_num_layers=2, viseme_num_classes=10, word_input_size=256, word_num_heads=4, word_hidden_size=512, word_num_layers=4, word_num_classes=100)

# Load the trained model weights
model_weights_path = "lip_reading_model.pth"
lip_reading_system.load_state_dict(torch.load(model_weights_path))
lip_reading_system.eval()

# Example usage
import torch

visual_input = torch.randn(1, 1, 3, 96, 128)
visual_input = visual_input.mean(dim=1)

if visual_input.shape[1] == 1:
    visual_input = torch.unsqueeze(visual_input, dim=1)

# Reshape the tensor to [1, 3, 32, 384] (which has 110592 elements)
#visual_input = visual_input.view(1, 3, 32, 384)

print(visual_input.size())
predicted_words = lip_reading_system(visual_input)
print(predicted_words)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lip_reading_system.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    lip_reading_system.train()
    running_loss = 0.0

    for i, (visual_input, viseme_label) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        predicted_visemes = lip_reading_system(visual_input)
        loss = criterion(predicted_visemes, viseme_label)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluation loop
lip_reading_system.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for visual_input, viseme_label in test_loader:
        predicted_visemes = lip_reading_system(visual_input)
        _, predicted_labels = torch.max(predicted_visemes, 1)
        total_correct += (predicted_labels == viseme_label).sum().item()
        total_samples += viseme_label.size(0)

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy * 100:.2f}%")



# Save the trained model to the specified directory
torch.save(lip_reading_system.state_dict(), "E:/mlops/Lip_Reading/RESEARCH PAPER_CODE/lip_reading_model.pth")


# Save the trained model 
torch.save(lip_reading_system.state_dict(), "lip_reading_model.pth")