from flask import Flask, render_template, request
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
PRED_FOLDER = "./predictions"
if not os.path.exists(PRED_FOLDER):
    os.makedirs(PRED_FOLDER)
app.config['PRED_FOLDER'] = PRED_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the model arch
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(conv)
        return conv, pool

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        diffZ = skip.size()[4] - x.size()[4]
        x = F.pad(x, [diffZ // 2, diffZ - diffZ // 2,
                      diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.out = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        x = self.out(x)
        return x

# Initialize the model
in_channels = 4  
out_channels = 1 
model = UNet3D(in_channels, out_channels)

MODEL_PATH = './model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')

        if len(files) != 4:
            return render_template('index.html', result="Please upload exactly 4 .nii files.")

        saved_files = []
        for file in files:
            if not file.filename.endswith('.nii'):
                return render_template('index.html', result="All files must be .nii format.")
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            saved_files.append(file_path)

        # Preprocess the files and stack into a single 4-channel tensor
        scans = []
        for file_path in saved_files:
            nii = nib.load(file_path)  
            data = nii.get_fdata()    
            data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalize
            scans.append(data)

        scans = np.stack(scans, axis=0)  
        scans = np.expand_dims(scans, axis=0)  
        input_tensor = torch.tensor(scans, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)

            output = output.squeeze().cpu()  

            output = (torch.sigmoid(output) > 0.5).float()
            
            output_np = (output.numpy() * 255).astype(np.uint8)
            output_np = output_np.squeeze()
            mask_img = Image.fromarray(output_np, mode="L")
            mask_img.save(mask_path)
        

        with torch.no_grad():
            mask_path = os.path.join(app.config['PRED_FOLDER'], "predicted_mask.png")
            output_image = output.squeeze()

            # Handle the case where the output has a batch dimension
            if len(output_image.shape) == 4:  # [B, C, H, W]
                output_image = output_image[0]  # Select the first batch item

            if len(output_image.shape) == 3:  # [C, H, W]
                # Select the middle slice along the depth (channel) dimension
                middle_slice = output_image[output_image.shape[0] // 2]
            elif len(output_image.shape) == 2:  # [H, W]
                middle_slice = output_image  # Already a single 2D image
            else:
                raise ValueError("Unexpected dimensions in output tensor.")

            # Plot and save the middle slice as a grayscale image
            plt.imshow(middle_slice, cmap='gray')
            plt.axis('off')  # Hide the axes
            plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)

            print(f"Mask saved at: {mask_path}")


        return render_template('index.html', result="Prediction completed", mask_path=mask_path)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)