## IV. ABOUT THE NETWORK MODEL

### A. Theory in Simple

For the task of radar-based HAR from DTMs, a novel neural network architecture is designed shown in Fig. 3. The core principle of this design is the replacement of the computationally intensive self-attention mechanisms commonly found in ViT with FFT-based global filter module. This module is engineered to facilitate the exchange of information across all spatial locations of the feature maps by operating in the frequency domain. The architecture is constructed to improve training stability and lightweight performance.

![FFTBasedNN](https://github.com/user-attachments/assets/8685098e-6d8a-4014-8b01-666a2ec88e6b)

Fig. 3. Structure of the proposed FFT-based neural network model.

### B. Codes Explanation (Folder: FFTBasedNN)

#### 1. FFTBasedNN.m ####

This MATLAB script implements the FFT-Based recognition model for radar HAR. It loads a dataset of DTM, constructs a custom neural network architecture incorporating FFT-based global filter layers, trains the model using specified options, and performs post-training analysis including visualization of accuracy/loss curves, positional embeddings, and learned global filters.

**Input:** Dataset directory ('dataset/') containing subdirectories for each class with DTM images.

**Output:** Trained model weights ('best_model.mat', 'final_model.mat'), console logs, 'curves.png' (training/validation curves), 'heatmaps.png' (positional embeddings), 'filter_map.png' (learned global filters).

#### 2. FFTBasedNN_Improved.py ####

This Python script using Paddlepaddle framework implements an improved version of the FFT-Based recognition model for radar HAR. It defines a custom dataset class, constructs the model with global filter layers, trains the model with label smoothing and AdamW optimizer, and visualizes training curves, positional embedding heatmaps, and learned global filters.

**Input:** Dataset directory ('dataset/') containing subdirectories for each class with DTM images.

**Output:** Trained model weights ('best_model.pdparams', 'final_model.pdparams'), console logs, 'curves.png' (training/validation curves), 'heatmaps.png' (positional embeddings), 'filter_map.png' (learned global filters).

#### 3. JoeyBG_FFTLayer.m ####

This MATLAB custom deep learning layer class defines the 'JoeyBG_FFTLayer', which implements the core global filtering operation in the frequency domain. It includes forward propagation using FFT, element-wise multiplication with learnable complex weights, and IFFT, as well as a custom backward function to handle complex gradients for compatibility with MATLAB's optimizers.

**Input:** Feature maps from previous layers; dimensions [Height, Width, Channels] specified during initialization.

**Output:** Filtered feature maps in the spatial domain with real part only.

### C. Datafiles Explanation (Folder: FFTBasedNN/dataset) ###

Store the image datasets for training and validation in the FFTBasedNN/dataset folder. Subfolders should be named after activity categories, with images placed directly inside each subfolder. This setup ensures the code runs correctly.
