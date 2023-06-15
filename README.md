# On the right food

## Introduction

"On the right food" is a web application created with Streamlit that can classify and segment different foods from real world images. It was our Capstone Project of the Data Science Bootcamp at neuefische.

We used a Convolutional Neural Network (CNN) to create a prediction model using the dataset from [AICrowd](https://link.to/thedataset).

A video of our presentation of the project is available [here](https://www.youtube.com/watch?v=ymSrVHMmX54).
The presentation can be downloaded [here](https://google_drive.com)

## 1. Prerequisites

1. Create the model

    For running the "On the right food" - food segmentation prediction web app you need to have a trained model. We created the [otfr-training repository](https://github.com/rhiniodontypus/otrf-training) to guide you how to preprocess your image files, create a corresponding annotations.json file and train the model.
    
2. Transfer the model to your local machine

    After the training is completed get the model file `model.pth` from the training `output` folder in the vm. For this you can use [file_transfer_vm.py](file_transfer_vm.py) by adding your GCP login credentials and the IP address of your virtual machine.

5. Place the trained `model.pth` file in the `./output` folder.

6. Place the `annotations.json` file in the `./annotations` folder.

7. If you use different names than the default ones, update the file names of your personal `model.pth` and `annotations.json` in the [settings.py](./config/settings.py).

## 2. Web App Installation

We recommend to set up a virtual environment. Make sure you use a pip version <= 23.0.1. Otherwise the installation of detectron2 will fail!

`python -m pip install --upgrade pip==23.0.1`


1. Install the required python packages:

    `python -m pip install -r requirements.txt`

2. Install detectron2:

    `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

## 3. Running the Web App
1. Open the web app in a terminal in your main repository path:

    `streamlit run main.py`

    The app should open automatically in your web browser.
2. Upload your image.