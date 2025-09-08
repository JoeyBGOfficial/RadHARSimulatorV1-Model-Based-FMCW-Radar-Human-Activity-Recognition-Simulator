# RadHARSimulator V1: Model-Based FMCW Radar Human Activity Recognition Simulator

## I. Introduction

![RadHARSimulator_Splash_Screen](https://github.com/user-attachments/assets/18dcaaab-0d71-4fc6-9c33-1a26986dd447)
Fig. 1. Splash screen of RadHARSimulator V1.

### Write Sth. Upfront:

From the very beginning of my research, I had planned to develop simulation software. This simulator consumed nearly a full year of my spare time. It involved countless hours of grueling debugging, but thankfully, I persevered and successfully made it.

I would like to thank my mentors for the platform they have provided me. I would also like to thank my fellow Xiaolong Sun and junior mate Jiarong Zhao. It was their encouragement that kept me going and enabled me to complete this work.

My software has not undergone extensive testing by a large number of users. There may still be areas for improvement during use. I welcome your valuable feedback and would be very grateful!

### Basic Information:

The V1 version of radar-based human activity recognition simulator (RadHARSimulator). This software provides a streamlined end-to-end simulation and analysis pipeline for FMCW radar human activity recognition. 12 activities, wall target, and all parameters of both radar system and human kinematic can be adjustable.

![Software_Interface](https://github.com/user-attachments/assets/93a5c747-6d71-4734-8c6a-d51a7510acef)

Fig. 2. The software interface of RadHARSimulator V1.

**My Email:** JoeyBG@126.com;

**Abstract:** Radar-based human activity recognition (HAR) is a pivotal research area for applications requiring non-invasive monitoring. However, the acquisition of diverse and high-fidelity radar datasets for robust algorithm development remains a significant challenge. To overcome this bottleneck, a model-based frequency-modulated continuous wave (FMCW) radar HAR simulator is developed. The simulator integrates an anthropometrically scaled $13$-scatterer kinematic model to simulate $12$ distinct activities. The FMCW radar echo model is employed, which incorporates dynamic radar cross-section (RCS), free-space or through-the-wall propagation, and a calibrated noise floor to ensure signal fidelity. The simulated raw data is then processed through a complete pipeline, including moving target indication (MTI), bulk Doppler compensation, and Savitzky-Golay denoising, culminating in the generation of high-resolution range-time map (RTM) and Doppler-time maps (DTMs) via both short-time Fourier transform (STFT) and Fourier synchrosqueezed transform (FSST). Finally, a novel neural network method is proposed to validate the effectiveness of the radar HAR. Numerical experiments demonstrate that the simulator successfully generates high-fidelity and distinct micro-Doppler signature, which provides a valuable tool for radar HAR algorithm design and validation.

**Corresponding Papers:**

[1]

### Notes:

**This simulator is currently available for installation only. If you require access to the source code, please contact the author to obtain permission.**

## II. How to Install

