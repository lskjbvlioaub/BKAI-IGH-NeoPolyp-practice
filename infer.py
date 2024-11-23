import torch
import os
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import segmentation_models_pytorch as smp
import argparse
import gdown
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
url = 'https://drive.google.com/uc?id=1OZm_9OSPnvAy8GlEjpTgn9Ama0bEYpPS'
output = 'colorization_model.pth' 
gdown.download(url, output, quiet=False)
'''
with open(output, 'rb') as f: 
    if f.read(4) == b'<htm': # Check if the file starts with an HTML tag 
        raise ValueError("Downloaded file is not a valid PyTorch model checkpoint. Please check the download link.")
'''
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)
model.to(device)

parser = argparse.ArgumentParser(description='Polyp Segmentation')
parser.add_argument('--image_path', type=str, help='Directory path to test images')
parser.add_argument('--predict_dir', type=str, default='/kaggle/working/bkai-igh-neopolyp-practice', help='Directory path to save output masks')
args = parser.parse_args()

checkpoint = torch.load(output, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

val_transform = A.Compose([A.PadIfNeeded(min_height=1024, min_width=1280, border_mode=0, value=0),
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

image = cv2.imread(args.image_path) 
#image = cv2.imread("/kaggle/working/bkai-igh-neopolyp-practice/7af2ed9fbb63b28163a745959c039830.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
#image = cv2.resize(image, (256, 256))
transformed = val_transform(image=image) 
input_image = transformed['image'].unsqueeze(0)
input_image = input_image.to(device)
with torch.no_grad(): output = model(input_image) 
_, predicted = torch.max(output, 1) 
predicted = predicted.squeeze(0).cpu().numpy()

color_dict= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}

colormap = np.array([
    [0, 0, 0],  # Class 0 - Black
    [255, 0, 0],  # Class 1 - Red
    [0, 255, 0]  # Class 2 - Green
])

# Apply the colormap
segmented_image = colormap[predicted].astype(np.uint8)
output_path = args.predict_dir + "\segmented_image.png" 
segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR) 
cv2.imwrite(output_path, segmented_image_bgr)
# Plot the original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.axis('off')

plt.show()
