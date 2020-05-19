# Handwritten Digits Recognition

Author: Kevin Ping-Sheng Lin

Date: May 17, 2020

## About this project

In this project, I am interested in building models that can recognize the digits written on the image inputs. I explored two Python libraries to perform the task, namely scikit-learn and TensorFlow. For scikit-learn, I explored three models — support vector machine (SVM), k nearest-neighbour (kNN) and random forest (RF); for TensorFlow, I constructed a convolutional neural network (CNN) for deep learning. The trained models are saved and the deployed on *DigitsVisualizer.py*.

The description of each file is listed below.
1. *Handwritten Digits Recognition 01 - Scikit-learn*: building three scikit-learn models and evaluate their accuracies
2. *Handwritten Digits Recognition 02 - TensorFlow*: building a large CNN to classify the images
3. *Applications*: two Python source codes that demonstrate practical deployments of trained models


## Experiemental results

The scikit-learn models and the CNN are trained on different databases, so their results are organized separately.

### Part 1: scikit-learn models

The scikit-learn models are trained on scikit-learn digits data set with 1797 images. From Table 1 below, we can see that RF is severely overfitted, as the difference in accuracies is around 3.5%. Then, since SVM and kNN are not overfitted, the last criteria to be compared is accuracy. SVM has higher than kNN in both accuracies. Thus, SVM is the best model out of three.

Table 1. Accuracies of scikit-learn models in recognizing handwritten digits
|          | Support vector machine | k nearest-neighbour | Random forest |
|----------|------------------------|---------------------|---------------|
| Training | 99.861 	              | 99.235 	            | 99.861        |
| Testing  | 99.167 	              | 98.889 	            | 96.389        |


### Part 2: TensorFlow CNN

The TensorFlow CNN is trained on the famous MNIST database, which contains 70,000 images. From Table 2, we see that while trained and tested using a large amount of images, the network achieves a splendid result of over 99% accuracy on both training and testing images. This is indeed a robust classification model!

Table 2. Accuracy of the TensorFlow model
|          | TensorFlow CNN |
|----------|----------------|
| Training | 99.46          |
| Testing  | 99.21 	        |

## Technologies used

* Operating System: Windows 10
* Programs: Anaconda, Jupyter Notebook
* Software: Python 3, scikit-learn, pandas, numpy, matplotlib, tensorflow, keras
