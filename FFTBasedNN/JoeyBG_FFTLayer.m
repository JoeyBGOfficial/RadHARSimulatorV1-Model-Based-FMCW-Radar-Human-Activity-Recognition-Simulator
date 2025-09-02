%% Custom Deep Learning Layer for FFT-Based Global Filtering
% Author: JoeyBG.
% Improved By: JoeyBG.
% Date: 2025.9.2.
% Platform: MATLAB R2025a.
% Affiliation: Beijing Institute of Technology.
%
% Information:
%   - This file defines a custom deep learning layer named 'JoeyBG_FFTLayer'.
%   - The layer implements the core logic of a Global Filter Network, which operates
%     in the frequency domain to achieve global information exchange, similar to a
%     self-attention mechanism but with lower computational cost.
%   - Key Operations:
%       1. 2D Fast Fourier Transform (FFT) to convert spatial feature maps to the frequency domain.
%       2. Element-wise multiplication with a learnable complex-valued filter.
%       3. 2D Inverse Fast Fourier Transform (IFFT) to convert the result back to the spatial domain.
%
% Notes:
%   - To make the complex-valued filter learnable within MATLAB's Deep Learning framework,
%       the weights are stored as a real-valued dlarray of size [H, W, C, 2], where the
%       last dimension separates the real and imaginary components.
%   - A custom 'backward' function is implemented. This is essential because MATLAB's
%       automatic differentiation engine produces complex gradients during backpropagation
%       for FFT-based operations, which are not compatible with standard optimizers like 'adam'.
%       This function manually calculates and returns real-valued gradients for both the
%       layer's input and its learnable weights.

%% Class Body
classdef JoeyBG_FFTLayer < nnet.layer.Layer
    properties (Learnable)
        % ComplexWeight: Learnable global filter.
        % Stored as a real-valued dlarray of size [Height, Width, Channels, 2].
        ComplexWeight
    end

    properties
        Height
        Width
        Channels
    end

    methods
        function layer = JoeyBG_FFTLayer(height, width, channels, name)
            % Constructor for the JoeyBG_FFTLayer.          
            % Assign layer name and description.
            layer.Name = name;
            layer.Description = 'Global Filter Layer based on FFT';
            
            % Store the dimensions of the feature maps this layer will process.
            layer.Height = height;
            layer.Width = width;
            layer.Channels = channels;
            
            % Initialize the learnable weights with small random values.
            layer.ComplexWeight = dlarray(randn(height, width, channels, 2) * 0.02);
        end

        function Z = predict(layer, X)
            % Forward pass function for the layer.            
            % 1. Convert the input feature map from the spatial domain to the frequency domain.
            F = fft2(X);
            
            % 2. Reconstruct the complex-valued filter from its stored real and imaginary parts.
            weightReal = layer.ComplexWeight(:, :, :, 1);
            weightImag = layer.ComplexWeight(:, :, :, 2);
            weight = complex(weightReal, weightImag);
            
            % 3. Apply the learnable global filter via element-wise multiplication.
            F_tilde = F .* weight;
            
            % 4. Convert the filtered features back to the spatial domain.
            Z_complex = ifft2(F_tilde);
            
            % 5. Return only the real part of the result to ensure compatibility with
            %    subsequent layers (e.g., ReLU, BatchNormalization) which expect real inputs.
            Z = real(Z_complex);
        end
        
        function [dLdX, dLdComplexWeight] = backward(layer, X, Z, dLdZ, memory)
            % Custom backward function to handle complex gradients during training.
            % --- Recompute variables from the forward pass for gradient calculation ---
            % 1. Convert input to frequency domain, same as in predict().
            F = fft2(X);

            % 2. Reconstruct the complex weight, same as in predict().
            weightReal = layer.ComplexWeight(:, :, :, 1);
            weightImag = layer.ComplexWeight(:, :, :, 2);
            weight = complex(weightReal, weightImag);

            % --- Backpropagation using the Chain Rule ---
            dLdY = dLdZ;
            dLdF_tilde = fft2(dLdY);
            grad_F_cplx = dLdF_tilde .* conj(weight);
            grad_w_cplx = dLdF_tilde .* conj(F);
            dLdX_cplx = ifft2(grad_F_cplx);
            
            % --- Format Gradients for the Optimizer ---
            dLdX = real(dLdX_cplx);
            sum_grad_w_cplx = sum(grad_w_cplx, 4);
            dLdWeightReal = real(sum_grad_w_cplx);
            dLdWeightImag = imag(sum_grad_w_cplx);
            dLdComplexWeight = cat(4, dLdWeightReal, dLdWeightImag);
        end
    end
end