# Car-model-classification-and-localization
Our model is built with 8000+ train images from 196 classes. It will detect the car model with bounding box(localization). 
Model is trained with MobileNet as base feature extractor and two paths were taken from it one for classification and another for bounding box.


## Dataset
Stanford Cars Dataset ⇨
https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset


1. Dataset contains around 8000+ train and 8000+ test images bwlonging to 196 classes
2. Bounding box information about the car object in the images is provided in the csv format

## Model Summary
![model_summary](https://github.com/aravind-etagi/car-model-classification-and-localization/blob/master/demo_files/mobile_net_model_image.png)

## Sample detection
Input Image                |  Predicted Image
:-------------------------:|:-------------------------:
![input_image](https://github.com/aravind-etagi/car-model-classification-and-localization/blob/master/demo_files/input_image.jpg)  |  ![predicted_image](https://github.com/aravind-etagi/car-model-classification-and-localization/blob/master/demo_files/predicted_image.jpeg)


## Deployed link
https://car-object-detection.herokuapp.com/

## Demo Video
App predicts the model of the car among 196 clases and plots the bounding box around the car of the given image

![demo gif](https://github.com/aravind-etagi/car-model-classification-and-localization/blob/master/demo_files/demo_gif.gif)

