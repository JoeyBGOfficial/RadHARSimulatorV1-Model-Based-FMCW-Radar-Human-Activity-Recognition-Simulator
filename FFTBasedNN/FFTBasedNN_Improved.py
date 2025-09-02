# -*- coding: utf-8 -*-
'''
Improved FFT-Based Recognition Model Designed for Radar Human Activity Recognition
'''
# Former Author: Lerbron.
# Improved By: JoeyBG.
# Date: 2025.9.1.
# Platform: Python 3.7, paddlepaddle 2.3.2.
# Affiliation: Beijing Institute of Technology.
# 
# Information:
#   - This work employs an improved neural network architecture combining FFT and Transformer 
#       to accomplish radar-based human activity recognition on Doppler-time maps (DTM). 
#       The network replaces the computationally intensive self-attention layer found in Vision Transformers 
#       with a Global Filter Layer. This layer utilizes three key operations: a 2D Discrete Fourier Transform (DFT) 
#       to convert features to the frequency domain, an element-wise multiplication with a set of learnable global filters, 
#       and an Inverse DFT to return the features to the spatial domain. This frequency domain approach allows 
#       for efficient information exchange across all spatial locations, offering a competitive 
#       and computationally less complex alternative to self-attention and MLP-based models.
#
#       The global filter layer primarily consists of three components: 2D FFT, learnable global filters, and 2D IFFT:
#       1. The 2D FFT uses the Fast Fourier Transform to convert an image from the spatial domain to the frequency domain, 
#           as expressed by the following formula:
#           $$\boldsymbol{X}=\mathcal{F}[\boldsymbol{x}] \in \mathbb{C}^{H \times W \times D}$$
#       2. Multiply the learnable global filter with the frequency domain features:
#           $$\tilde{\boldsymbol{X}}=\boldsymbol{K} \odot \boldsymbol{X}$$
#       3. 2D IFFT uses the inverse fast Fourier transform to convert the image from the frequency domain back to the spatial domain:
#           $$\boldsymbol{x} \leftarrow \mathcal{F}^{-1}[\tilde{\boldsymbol{X}}]$$
#
# Input:
#   - A dataset directory ('dataset/') containing subdirectories for each class.
#   - Each subdirectory should be named after its class and contain the corresponding
#     Doppler-Time Map (DTM) images (e.g., PNG, JPG).
#
# Output:
#   - Trained model weights ('best_model.pdparams', 'final_model.pdparams') and optimizer states
#     saved in the 'work/model/' directory.
#   - Console logs detailing the training and validation progress for each epoch.
#   - 'curves.png': A plot showing the training and validation accuracy/loss curves over epochs.
#   - 'heatmaps.png': A visualization of the model's learned positional embeddings.
#   - 'filter_map.png': A 12x12 grid visualizing the learned global filters from the model's layers.
#
# Notes:
#   - The script is configured to run on 'gpu:0' by default. This can be changed to 'cpu'.
#   - For an unknown reason, the 'cv2' library must be imported for the training to proceed without errors.
#   - The model architecture can be switched between 'tiny', 'XS', 'S', and 'B' versions by commenting/uncommenting
#     the relevant lines in the 'Construction of the Models' and 'Training and Validation' sections.
#   - Data augmentation techniques like random flipping are intentionally disabled as they may not be
#     suitable for radar DTM data.
#   - The model uses a Label Smoothing Cross-Entropy loss function for potentially better generalization.
#
# References:
#   [1] Rao, Y., Zhao, W., & Liu, Z., et al. (2021). Global Filter Networks for Image Classification.
#       arXiv preprint arXiv:2107.00645.
#   [2] Dosovitskiy, A., Beyer, L., & Kolesnikov, A., et al. (2020). An Image is Worth 16x16 Words:
#       Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929.
#   [3] Huang, G., Sun, Y., & Liu, Z., et al. (2016). Deep Networks with Stochastic Depth.
#       European conference on computer vision (ECCV).

'''
Preparations for the Script 
'''
# Import necessary libraries for code running.
# %matplotlib inline
import paddle
import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt
from paddle.vision.datasets import Cifar10
from paddle.vision.transforms import Compose, Normalize, Transpose, Resize, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter
from paddle.io import Dataset, DataLoader
from paddle import nn
from paddle.nn import CrossEntropyLoss
import paddle.nn.functional as F
import paddle.vision.transforms as transforms
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import itertools
import random
import math
import cv2 # For reasons that are unclear, network training does not work if the cv2 library is not introduced.
import time
from sklearn.metrics import confusion_matrix

# Initialization of the Python script.
print("---------- Author: © JoeyBG © ----------")
# Execution the training with GPU of number 0.
paddle.device.set_device('gpu:0')
# Excution the training with CPU.
# paddle.device.set_device('cpu')
# Set the backend of the code running to cv2 image format.
paddle.vision.set_image_backend('cv2')

'''
Parameter Definition
'''
# Path definition.
data_dir = 'dataset' # Define the path to your dataset.
work_path = 'work/model' # Define the path for saving model.

# Training parameters.
learning_rate = 0.00147 # Learning rate for training.
n_epochs = 20 # Number of epochs for training.
train_ratio = 0.8 # Ratio of training data to total dataset.
batch_size = 256 # Batch size predefined for training and validation dataloader.
num_classes = 12 # Number of classes in the dataset.
embed_dim_tiny = 256 # Embedding dimension of the tiny version of the network.
embed_dim_XS = 384 # Embedding dimension of the XS version of the network.
embed_dim_S = 384 # Embedding dimension of the S version of the network.
embed_dim_B = 512 # Embedding dimension of the B version of the network.
depth_tiny = 12 # Depth of the tiny version of the network.
depth_XS = 12 # Depth of the XS version of the network.
depth_S = 19 # Depth of the S version of the network.
depth_B = 19 # Depth of the B version of the network.
# paddle.seed(42) # Set random seed for reproducibility.
# np.random.seed(42)

'''
Data Augmentation and Normalization
'''
# Using data augmentation and normalization during training.
# train_tfm = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
#     transforms.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.2),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.RandomRotation(20),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
# During testing, we only need to resize and normalize the input image.
# test_tfm = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])
# Define the uniform transformations for both training and validation data augmentation.
Estimation_Resolution = 224 # Define the resolution of input images. This parameter can be found in MATLAB feature augmentation script.
transform = transforms.Compose([
    transforms.Resize((Estimation_Resolution, Estimation_Resolution)),
    # Considering the specificity of radar images, here we may not use random contrast data augmentation.
    # ColorJitter(0.4, 0.4, 0.4), 
    # We are not suggesting to mix up images in validation or testing sets.
    # paddlex.transforms.MixupImage(),
    # Highly not recommended to do random flip in data augmentation.
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

'''
Dataset Construction
'''
# Define a custom dataset class.
class ImageDataset(paddle.io.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

# Read the training and validation dataset and split it into two.
dataset = []
label_list = sorted(os.listdir(data_dir))
for label_idx, label in enumerate(label_list):
    label_dir = os.path.join(data_dir, label)
    image_list = os.listdir(label_dir)
    random.shuffle(image_list)
    num_train = int(len(image_list) * train_ratio)
    dataset.extend([(os.path.join(label_dir, img), label_idx) for img in image_list])

# Create an instance of the custom dataset class.
full_dataset = ImageDataset(dataset, transform=transform)

# Calculate the number of samples for train and validation sets.
num_train_samples = int(len(full_dataset) * train_ratio)
num_val_samples = len(full_dataset) - num_train_samples

# Split the dataset into train and validation sets.
train_dataset, val_dataset = paddle.io.random_split(full_dataset, lengths=[num_train_samples, num_val_samples])

# Create data loaders for train and validation sets.
train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("Number of Datas in Training Set: %d" % len(train_dataset))
print("Number of Datas in Validation Set: %d" % len(val_dataset))

'''
Network Model Definition
'''
# Definition of the label-smoothed cross-entropy loss function.
class LabelSmoothingCrossEntropy(nn.Layer):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):

        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(pred, axis=-1)
        idx = paddle.stack([paddle.arange(log_probs.shape[0]), target], axis=1)
        nll_loss = paddle.gather_nd(-log_probs, index=idx)
        smooth_loss = paddle.mean(-log_probs, axis=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()
    
# Standard drop path function.
def drop_path(x, drop_prob=0.0, training=False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output

# Define of the drop path.
class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
# Define the MLP module of the network.
class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Define the global filter module of the network.
class GlobalFilter(nn.Layer):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = self.create_parameter(attr=None, shape=(h, w, dim, 2),
            dtype='float32', is_bias=False,
            default_initializer=nn.initializer.Assign(paddle.randn(shape=(h, w, dim, 2), dtype='float32') * 0.02))
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.reshape((B, a, b, C))

        x = paddle.to_tensor(x, paddle.float32)

        x = paddle.fft.rfft2(x, axes=(1, 2), norm='ortho')
        weight = paddle.as_complex(self.complex_weight)
        x = x * weight
        x = paddle.fft.irfft2(x, s=(a, b), axes=(1, 2), norm='ortho')

        x = x.reshape((B, N, C))

        return x

# Define the improved FFT-based attention module of the network.
class Block(nn.Layer):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x
    
# Define the class structure of GFNet.
class GFNet(nn.Layer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None,
                 dropcls=0):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or nn.LayerNorm

        assert img_size % patch_size == 0, 'Image size cannot be divided by patch size'

        self.patch_embed = nn.Conv2D(in_chans, embed_dim, patch_size, patch_size)
        num_patches = (img_size//patch_size) * (img_size//patch_size)

        self.pos_embed = self.create_parameter(attr=None, shape=(1, num_patches, embed_dim),
            dtype='float32', is_bias=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
           )
        else:
            self.pre_logits = nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        init = paddle.nn.initializer.TruncatedNormal(mean=0.0, std=.02)
        init(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        zeros_ = nn.initializer.Constant(value=0.)
        ones_ = nn.initializer.Constant(value=1.)
        if isinstance(m, (nn.Linear, nn.Conv2D)):
            init = paddle.nn.initializer.TruncatedNormal(mean=0.0, std=.02)
            init(m.weight)
            if isinstance(m, (nn.Linear, nn.Conv2D)) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            zeros_(m.bias)
            ones_(m.weight)


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = paddle.flatten(x, 2).transpose([0, 2, 1])
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x

'''
Construction of the Models
'''
# Using tiny version of GFNet for Network model construction.
model_tiny = GFNet(num_classes=num_classes, embed_dim=embed_dim_tiny, depth=depth_tiny)
paddle.summary(model_tiny, (batch_size, 3, 224, 224))

# # Using XS version of GFNet for Network model construction.
# model_XS = GFNet(num_classes=num_classes, embed_dim=embed_dim_XS, depth=depth_XS)
# paddle.summary(model_XS, (batch_size, 3, 224, 224))

# # Using S version of GFNet for Network model construction.
# model_S = GFNet(num_classes=num_classes, embed_dim=embed_dim_S, depth=depth_S)
# paddle.summary(model_S, (batch_size, 3, 224, 224))

# # Using B version of GFNet for Network model construction.
# model_B = GFNet(num_classes=num_classes, embed_dim=embed_dim_B, depth=depth_B)
# paddle.summary(model_B, (batch_size, 3, 224, 224))

'''
Training and Validation
'''
# Create the directory if it does not exist.
if not os.path.exists(work_path):
    os.makedirs(work_path)

# Assign the pre-defined model for training.
model = model_tiny
# model = model_XS
# model = model_S
# model = model_B

# Initialize the loss function.
criterion = LabelSmoothingCrossEntropy()

# Configure the AdamW optimizer with a learning rate scheduler and gradient clipping.
grad_norm = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=learning_rate, T_max=len(train_loader) * n_epochs, eta_min=1e-5, verbose=False)
optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=scheduler, weight_decay=0.05, grad_clip=grad_norm)

# Initialize lists to store metrics for plotting.
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# Keep track of the best validation accuracy.
best_acc = 0.0

print("[INFO] Starting training process...")

# Loop over the specified number of epochs.
for epoch in range(n_epochs):
    # Set the model to training mode.
    model.train()
    
    # Initialize accumulators for epoch-wide training metrics.
    train_loss_epoch = 0.0
    train_num = 0
    train_accuracy_manager = paddle.metric.Accuracy()
    
    print(f"[INFO] Epoch: {epoch+1}/{n_epochs} | Learning Rate: {optimizer.get_lr():.8f}")
    
    # Iterate over batches of training data.
    for batch_id, data in enumerate(train_loader):
        x_data, y_data = data
        labels = paddle.unsqueeze(y_data, axis=1)

        # Perform the forward pass.
        logits = model(x_data)
        # Calculate the loss.
        loss = criterion(logits, y_data)

        # Perform backpropagation and update model weights.
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Clear gradients for the next iteration.
        optimizer.clear_grad()
        
        # Calculate accuracy for the current batch.
        acc = paddle.metric.accuracy(logits, labels)

        # Accumulate training loss and accuracy for the epoch.
        train_accuracy_manager.update(acc)
        train_loss_epoch += loss.item() * len(y_data)
        train_num += len(y_data)
        
        # This print statement remains to show per-batch progress.
        print(f"[TRAIN] Epoch: {epoch+1}/{n_epochs} | Batch: {batch_id+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc.item()*100:.2f}%")
    
    # Calculate and store average training metrics for the epoch.
    avg_train_loss = train_loss_epoch / train_num
    avg_train_acc = train_accuracy_manager.accumulate()
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    print() # Newline after finishing the epoch's training batches.

    # Set the model to evaluation mode.
    model.eval()
    val_loss = 0.0
    val_num = 0
    val_accuracy_manager = paddle.metric.Accuracy()

    # Disable gradient calculations for validation.
    with paddle.no_grad():
        # Iterate over batches of validation data.
        for batch_id, data in enumerate(val_loader):
            x_data, y_data = data
            labels = paddle.unsqueeze(y_data, axis=1)
            
            logits = model(x_data)
            loss = criterion(logits, y_data)

            val_accuracy_manager.update(paddle.metric.accuracy(logits, labels))

            val_loss += loss.item() * len(y_data)
            val_num += len(y_data)

    # Calculate final validation metrics for the epoch.
    total_val_loss = val_loss / val_num
    val_acc = val_accuracy_manager.accumulate()
    
    # Store validation metrics for the epoch.
    history['val_loss'].append(total_val_loss)
    history['val_acc'].append(val_acc)

    print(f"[VALID] Epoch: {epoch+1}/{n_epochs} | Val Loss: {total_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    # Check if the current model is the best one.
    if val_acc > best_acc:
        best_acc = val_acc
        print(f"[INFO] New best validation accuracy: {best_acc*100:.2f}%. Saving model...")
        # Save the model state and optimizer state.
        paddle.save(model.state_dict(), os.path.join(work_path, 'best_model.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(work_path, 'best_optimizer.pdopt'))
    
    print("-" * 64) 

print(f"[INFO] Training finished. Best validation accuracy: {best_acc*100:.2f}%")
# Save the final model state at the end of training.
print(f"[INFO] Saving final model state...")
paddle.save(model.state_dict(), os.path.join(work_path, 'final_model.pdparams'))
paddle.save(optimizer.state_dict(), os.path.join(work_path, 'final_optimizer.pdopt'))
print("[INFO] Final model saved successfully.")

'''
Visualization of the Curves
'''
# Create a range of epochs for the x-axis.
epochs_range = range(n_epochs)

# Create a figure with two subplots, one for accuracy and one for loss.
plt.figure(figsize=(14, 6))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_acc'], label='Training Accuracy', marker='o')
plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Subplot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_loss'], label='Training Loss', marker='o')
plt.plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

# Display the plots.
plt.suptitle('Model Training History', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('curves.png', dpi=300, bbox_inches='tight')
plt.show()

'''
Visualization of the Heatmap
'''
# Squeeze the tensor to remove the batch dimension.
pos_embed_tensor = model.pos_embed.squeeze() 

# Reshape the tensor into a 3D format (height, width, channels) and then transpose it to (channels, height, width).
ans = pos_embed_tensor.reshape((14, 14, 256)).transpose([2, 0, 1]) # Corrected shape: [256, 14, 14]

# Create a 2x2 grid of subplots to display multiple heatmaps in a single figure.
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Visualization of Positional Embedding Heatmaps', fontsize=16)

# Display the heatmap of the mean of all channels.
im1 = axes[0, 0].imshow(ans.mean(axis=0).numpy())
axes[0, 0].set_title('Mean of All Channels')
# Add a color bar to the subplot to indicate the scale.
fig.colorbar(im1, ax=axes[0, 0])

# Display the heatmap for the first channel (index 0).
im2 = axes[0, 1].imshow(ans[0, :, :].numpy())
axes[0, 1].set_title('Channel 0')
fig.colorbar(im2, ax=axes[0, 1])

# Display the heatmap for a middle channel (index 127).
im3 = axes[1, 0].imshow(ans[127, :, :].numpy())
axes[1, 0].set_title('Channel 127')
fig.colorbar(im3, ax=axes[1, 0])

# Display the heatmap for the last channel (index 255).
im4 = axes[1, 1].imshow(ans[255, :, :].numpy())
axes[1, 1].set_title('Last Channel (255)')
fig.colorbar(im4, ax=axes[1, 1])

# Adjust the layout to prevent titles and labels from overlapping.
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('heatmaps.png', dpi=300, bbox_inches='tight')
# Render and display the plot.
plt.show()

'''
Visualization of the Filters
'''
# Assume 'model' is your trained and loaded PaddlePaddle model object.
model.eval() 
model_state_dict = model.state_dict()


def show_single_filter(gf):
    # Processes a single global filter from the frequency domain to the spatial domain for visualization.
    h = gf.shape[0]
    gf_complex = paddle.as_complex(gf)
    gf_spatial = paddle.fft.irfft2(gf_complex, axes=(0, 1), s=(h, h))
    gf_complex = paddle.fft.fft2(gf_spatial, axes=(0, 1))
    gf_complex = paddle.fft.fftshift(gf_complex, axes=(0, 1))
    gf_abs = gf_complex.abs()
    return gf_abs

n_viz_channel = 12
global_filters = []

for i_layer in range(12):
    weight = model_state_dict[f'blocks.{i_layer}.filter.complex_weight']
    for i_channel in range(n_viz_channel):
        global_filters.append(show_single_filter(weight[:, :, i_channel])[None])

global_filters = paddle.concat(global_filters)
print(f"Shape of the final filters tensor: {global_filters.shape}")

# Create a figure to hold the grid of filter visualizations.
fig = plt.figure(figsize=(12, 12))
rows = 12
cols = 12

# Loop to create a 12x12 grid of images.
for i in range(1, rows * cols + 1):
    img_array = global_filters[i-1].squeeze().numpy()
    ax = fig.add_subplot(rows, cols, i)
    plt.axis('off') 
    plt.imshow(img_array, cmap='YlGnBu')

# Adjust spacing between subplots to make the layout less compact.
# Original values were (wspace=0, hspace=0). We increase them to add padding.
plt.subplots_adjust(wspace=0.1, hspace=0.1) 

# Save the final figure to a file.
plt.savefig('filter_map.png', dpi=300, bbox_inches='tight')
# Display the plot.
plt.show()