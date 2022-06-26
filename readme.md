# Project Name
Sign Language Translator

This project uses a USB camera and detects what letter you are holding up. You can output the letter shown on the screen.

[Imgur](https://i.imgur.com/40Xp6x7.jpg)

## The Algorithm

The project used the Sign Language MNIST dataset to train and verify. Step 1 is to install venv and use the virtual enviornment. step_2_dataset.py sets up the dataset using pytorch. step_3_train.py trains the machine based on the provided dataset of images and connected labels. step_4_evalutate prints the accuracy of the machine and exports it as an onnx file. step_5_camera runs the onnx file and shows the camera. Output is eventually printed to output.txt.

## Running this project
In nvidia-project/sign-language-translator
Download venv using: python3 -m venv sign-language-translator
Run: source sign-language-translator/bin/activate , to activate virtual enviornment
Change directories into src
Using apt-get, install libsm6, libxext6, libxrender-dev (sudo apt-get install libsm6 libxext6 libxrender-dev)
Using pip install torch, torchvision, opencv-python, numpy, onnx, onnxruntime
Run: python step_5_camera.py

https://youtube.com/shorts/yilQClHC-E8?feature=share
