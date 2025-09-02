%% FFT-Based Recognition Model Designed for Radar Human Activity Recognition
% Former Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025.9.2.
% Platform: MATLAB R2025a.
% Affiliation: Beijing Institute of Technology.
% 
% Information:
%   - This work employs an improved neural network architecture combining FFT and Transformer 
%       to accomplish radar-based human activity recognition on Doppler-time maps (DTM). 
%       The network replaces the computationally intensive self-attention layer found in Vision Transformers 
%       with a Global Filter Layer. This layer utilizes three key operations: a 2D Discrete Fourier Transform (DFT) 
%       to convert features to the frequency domain, an element-wise multiplication with a set of learnable global filters, 
%       and an Inverse DFT to return the features to the spatial domain. This frequency domain approach allows 
%       for efficient information exchange across all spatial locations, offering a competitive 
%       and computationally less complex alternative to self-attention and MLP-based models.
%
%       The global filter layer primarily consists of three components: 2D FFT, learnable global filters, and 2D IFFT:
%       1. The 2D FFT uses the Fast Fourier Transform to convert an image from the spatial domain to the frequency domain, 
%           as expressed by the following formula:
%           $$\boldsymbol{X}=\mathcal{F}[\boldsymbol{x}] \in \mathbb{C}^{H \times W \times D}$$
%       2. Multiply the learnable global filter with the frequency domain features:
%           $$\tilde{\boldsymbol{X}}=\boldsymbol{K} \odot \boldsymbol{X}$$
%       3. 2D IFFT uses the inverse fast Fourier transform to convert the image from the frequency domain back to the spatial domain:
%           $$\boldsymbol{x} \leftarrow \mathcal{F}^{-1}[\tilde{\boldsymbol{X}}]$$
%
% Input:
%   - A dataset directory ('dataset/') containing subdirectories for each class.
%   - Each subdirectory should be named after its class and contain the corresponding
%     Doppler-Time Map (DTM) images (e.g., PNG, JPG).
%
% Output:
%   - Trained model weights ('best_model.mat', 'final_model.mat') saved in the 'work/model/' directory.
%   - Console logs detailing the training and validation progress for each epoch.
%   - 'curves.png': A plot showing the training and validation accuracy/loss curves over epochs.
%   - 'heatmaps.png': A visualization of the model's learned positional embeddings.
%   - 'filter_map.png': A 12x12 grid visualizing the learned global filters from the model's layers.
%
% Notes:
%   - The script is configured to run on GPU if available. This can be changed to CPU.
%   - The model architecture is set to 'tiny' version.
%   - Data augmentation techniques like random flipping are intentionally disabled as they may not be
%     suitable for radar DTM data.
%   - The model uses a Label Smoothing Cross-Entropy loss function for potentially better generalization.
%
% References:
%   [1] Rao, Y., Zhao, W., & Liu, Z., et al. (2021). Global Filter Networks for Image Classification.
%       arXiv preprint arXiv:2107.00645.
%   [2] Dosovitskiy, A., Beyer, L., & Kolesnikov, A., et al. (2020). An Image is Worth 16x16 Words:
%       Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929.
%   [3] Huang, G., Sun, Y., & Liu, Z., et al. (2016). Deep Networks with Stochastic Depth.
%       European conference on computer vision (ECCV).

%% Initialization of Matlab Script
clear all;
close all;
clc;
disp("---------- © Author: JoeyBG © ----------");
JoeyBG_Colormap = [ 0.6196	0.0039	0.2588
                    0.8353	0.2431	0.3098
                    0.9569	0.4275	0.2627
                    0.9922	0.6824	0.3804
                    0.9961	0.8784	0.5451
                    1.0000	1.0000	0.7490
                    0.9020	0.9608	0.5961
                    0.6706	0.8667	0.6431
                    0.4000	0.7608	0.6471
                    0.1961	0.5333	0.7412
                    0.3686	0.3098	0.6353 ]; % My favorite colormap.
JoeyBG_Colormap_Flip = flip(JoeyBG_Colormap);

% Display parameters
Font_Name = 'Palatino Linotype';
Font_Size_Basis = 12;
Font_Size_Axis = 13;
Font_Size_Title = 15;
Font_Weight_Basis = "normal";
Font_Weight_Axis = "normal";
Font_Weight_Title = "bold";

%% Import Data
% Import training and validation data.
% Read data from file: Corner Dataset/ as an example.
% The function imageDatastore automatically divides and labels data sets by file name.
Dataset_Path = "dataset"; % Change the data path to your own dataset.
if Dataset_Path == " "
    error("Dataset_Path is empty! Change it to your own path where dataset stores.")
end
imdsTrain = imageDatastore(Dataset_Path,"IncludeSubfolders",true,"LabelSource","foldernames"); % Inport dataset.
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.8,"randomized"); % The number of training data:the number of validation data = 4:1.

% Resize the images to match the network input layer.
% The input image should be in 3 channels but whatever spatial size you want.
% However, we strongly suggest the spatial size to be 2^n format.
augimdsTrain = augmentedImageDatastore([256 256 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([256 256 3],imdsValidation);

%% Set Training Options
% Specify options to use when training.
% The following are just the suggestion of the training options:
% Optimizer: Adam.
% Execution environment: GPU or CPU, choose automatically.
% Initial learning rate: 0.00147.
% Training epoches: 20.
% Batch size: 64.
% Shuffle frequency: per epoch.
% Validation frequency: per 20 batches.
% Plot: training progress.
% Validation data: from the variable "augimdsValidation" defined above.
% Validation output: Best epoch.
% Learning rate schadule: Piecewise.
opts = trainingOptions("adam",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.00147,...
    "MaxEpochs",20,...
    "MiniBatchSize",64,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",20,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation, ...
    "OutputNetwork","best-validation",...
    "LearnRateSchedule","piecewise");

%% Create Layer Graph
% Create the layer graph variable to contain the network layers.
lgraph = layerGraph();

%% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear array of layers.
% These are the layers with convolution operations and residual links.
tempLayers = [
    imageInputLayer([256 256 3],"Name","imageinput")
    convolution2dLayer([3 3],64,"Name","conv","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    maxPooling2dLayer([3 3],"Name","maxpool","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(3,"Name","addition")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(3,"Name","addition_1")
    reluLayer("Name","relu_6")
    convolution2dLayer([3 3],128,"Name","conv_5","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(3,"Name","addition_2")
    reluLayer("Name","relu_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(3,"Name","addition_3")
    reluLayer("Name","relu_13")
    convolution2dLayer([3 3],256,"Name","conv_10","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_14")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_16")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(3,"Name","addition_4")
    reluLayer("Name","relu_17")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_13")
    reluLayer("Name","relu_18")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_14")
    reluLayer("Name","relu_19")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(3,"Name","addition_5")
    reluLayer("Name","relu_20")
    convolution2dLayer([3 3],512,"Name","conv_15","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_15")
    reluLayer("Name","relu_21")
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(12,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% Add the custom FFT-based layers for each network block.
% The dimensions (Height, Width, Channels) are set according to the feature map size at that point in the network.
FFT_Layer_0 = JoeyBG_FFTLayer(64, 64, 64, "JoeyBG_FFT_Layer_0");
FFT_Layer_1 = JoeyBG_FFTLayer(64, 64, 64, "JoeyBG_FFT_Layer_1");
FFT_Layer_2 = JoeyBG_FFTLayer(32, 32, 128, "JoeyBG_FFT_Layer_2");
FFT_Layer_3 = JoeyBG_FFTLayer(32, 32, 128, "JoeyBG_FFT_Layer_3");
FFT_Layer_4 = JoeyBG_FFTLayer(16, 16, 256, "JoeyBG_FFT_Layer_4");
FFT_Layer_5 = JoeyBG_FFTLayer(16, 16, 256, "JoeyBG_FFT_Layer_5");
lgraph = addLayers(lgraph, FFT_Layer_0);
lgraph = addLayers(lgraph, FFT_Layer_1);
lgraph = addLayers(lgraph, FFT_Layer_2);
lgraph = addLayers(lgraph, FFT_Layer_3);
lgraph = addLayers(lgraph, FFT_Layer_4);
lgraph = addLayers(lgraph, FFT_Layer_5);

% Clean up helper variable.
clear tempLayers;

%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.
% Add all the layer variables defined above to the lgraph() container in the order in which they will be processed by the network.

% -- Connections for Block 0 (Source: maxpool, Target: addition) --
lgraph = connectLayers(lgraph,"maxpool","conv_1");
lgraph = connectLayers(lgraph,"maxpool","conv_2");
lgraph = connectLayers(lgraph,"maxpool","JoeyBG_FFT_Layer_0"); % FFT branch
lgraph = connectLayers(lgraph,"JoeyBG_FFT_Layer_0","addition/in1");
lgraph = connectLayers(lgraph,"relu_1","multiplication/in1");
lgraph = connectLayers(lgraph,"relu_2","multiplication/in2");
lgraph = connectLayers(lgraph,"multiplication","addition/in2");
lgraph = connectLayers(lgraph,"maxpool","addition/in3"); % Residual branch

% -- Connections for Block 1 (Source: relu_3, Target: addition_1) --
lgraph = connectLayers(lgraph,"relu_3","conv_3");
lgraph = connectLayers(lgraph,"relu_3","conv_4");
lgraph = connectLayers(lgraph,"relu_3","addition_1/in2"); % Original residual branch
lgraph = connectLayers(lgraph,"relu_4","multiplication_1/in1");
lgraph = connectLayers(lgraph,"relu_5","multiplication_1/in2");
lgraph = connectLayers(lgraph,"multiplication_1","addition_1/in1");
lgraph = connectLayers(lgraph,"relu_3","JoeyBG_FFT_Layer_1");
lgraph = connectLayers(lgraph,"JoeyBG_FFT_Layer_1","addition_1/in3");

% -- Connections for Block 2 (Source: relu_7, Target: addition_2) --
lgraph = connectLayers(lgraph,"relu_7","conv_6");
lgraph = connectLayers(lgraph,"relu_7","conv_7");
lgraph = connectLayers(lgraph,"relu_7","addition_2/in2"); % Original residual branch
lgraph = connectLayers(lgraph,"relu_8","multiplication_2/in1");
lgraph = connectLayers(lgraph,"relu_9","multiplication_2/in2");
lgraph = connectLayers(lgraph,"multiplication_2","addition_2/in1");
lgraph = connectLayers(lgraph,"relu_7","JoeyBG_FFT_Layer_2");
lgraph = connectLayers(lgraph,"JoeyBG_FFT_Layer_2","addition_2/in3");

% -- Connections for Block 3 (Source: relu_10, Target: addition_3) --
lgraph = connectLayers(lgraph,"relu_10","conv_8");
lgraph = connectLayers(lgraph,"relu_10","conv_9");
lgraph = connectLayers(lgraph,"relu_10","addition_3/in2"); % Original residual branch
lgraph = connectLayers(lgraph,"relu_11","multiplication_3/in1");
lgraph = connectLayers(lgraph,"relu_12","multiplication_3/in2");
lgraph = connectLayers(lgraph,"multiplication_3","addition_3/in1");
lgraph = connectLayers(lgraph,"relu_10","JoeyBG_FFT_Layer_3");
lgraph = connectLayers(lgraph,"JoeyBG_FFT_Layer_3","addition_3/in3");

% -- Connections for Block 4 (Source: relu_14, Target: addition_4) --
lgraph = connectLayers(lgraph,"relu_14","conv_11");
lgraph = connectLayers(lgraph,"relu_14","conv_12");
lgraph = connectLayers(lgraph,"relu_14","addition_4/in2"); % Original residual branch
lgraph = connectLayers(lgraph,"relu_15","multiplication_4/in1");
lgraph = connectLayers(lgraph,"relu_16","multiplication_4/in2");
lgraph = connectLayers(lgraph,"multiplication_4","addition_4/in1");
lgraph = connectLayers(lgraph,"relu_14","JoeyBG_FFT_Layer_4");
lgraph = connectLayers(lgraph,"JoeyBG_FFT_Layer_4","addition_4/in3");

% -- Connections for Block 5 (Source: relu_17, Target: addition_5) --
lgraph = connectLayers(lgraph,"relu_17","conv_13");
lgraph = connectLayers(lgraph,"relu_17","conv_14");
lgraph = connectLayers(lgraph,"relu_17","addition_5/in2"); % Original residual branch
lgraph = connectLayers(lgraph,"relu_18","multiplication_5/in1");
lgraph = connectLayers(lgraph,"relu_19","multiplication_5/in2");
lgraph = connectLayers(lgraph,"multiplication_5","addition_5/in1");
lgraph = connectLayers(lgraph,"relu_17","JoeyBG_FFT_Layer_5");
lgraph = connectLayers(lgraph,"JoeyBG_FFT_Layer_5","addition_5/in3");

% Display and analyze the network structure we've constructed above.
% plot(lgraph);
analyzeNetwork(lgraph);

%% Train Network
% Train the network using the specified options and training data.
[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);

%% Post-Training Analysis and Visualization
% Use the trained network to classify the validation dataset.
[YPred, probs] = classify(net, augimdsValidation);
YValidation = imdsValidation.Labels;

% Create a new figure for the confusion matrix.
figure('Name','Validation Set Confusion Matrix');
cm = confusionchart(YValidation, YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.FontName = Font_Name;
cm.FontSize = Font_Size_Basis;

% Sort the classes for consistent visualization.
class_names = categories(imdsTrain.Labels);
sortClasses(cm, class_names);

% Create a new figure to display the filter heatmaps.
figure('Name', 'Learned Global Filter Magnitudes', 'Color', 'white');
sgtitle('Heatmaps of Learned Global Filter Magnitudes (Averaged Across Channels)', ...
    'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
subplot_idx = 1;

% Loop through all layers of the trained network to find the custom FFT layers.
for i = 1:length(net.Layers)
    % Check if the current layer is an instance of JoeyBG_FFTLayer.
    if isa(net.Layers(i), 'JoeyBG_FFTLayer')
        layer = net.Layers(i);
        
        % Create a 2x3 grid of subplots for the 6 FFT layers.
        subplot(2, 3, subplot_idx);
        
        % Extract the learnable weights (ComplexWeight).
        weights = layer.ComplexWeight;
        real_part = weights(:,:,:,1);
        imag_part = weights(:,:,:,2);
        magnitude = sqrt(real_part.^2 + imag_part.^2);
        avg_magnitude = mean(magnitude, 3);
        filter_map_to_plot = fftshift(gather(avg_magnitude));
        filter_map_to_plot = filter_map_to_plot';
        
        % Display the filter map as an image.
        imagesc(filter_map_to_plot);
        colormap(gca, JoeyBG_Colormap_Flip);
        colorbar;
        axis off;
        axis equal tight;
        title(layer.Name, 'Interpreter', 'none', 'FontName', Font_Name, ...
              'FontSize', Font_Size_Axis, 'FontWeight', Font_Weight_Axis);
        
        % Increment the subplot index for the next layer.
        subplot_idx = subplot_idx + 1;
    end
end