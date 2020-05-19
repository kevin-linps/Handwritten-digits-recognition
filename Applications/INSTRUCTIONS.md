# Instructions on using the GUI

This folder contains two binaries and two Python source codes. The binaries are the trained models from the Jupyter notebooks in the main folder. The two Python source codes are applications using the binaries.

## *digit_visualization.py*

This is a visualizer for the trained models. It loads random images from the selected data set. Then, it displays the image along with the model's prediction and the actual answer.

Instructions:
1. From the drop-down menu, select the database you would like to choose the image from.
2. Click on the button on the top-right that says "Choose an image".
3. An image would then appear. (Note: When selecting "MNIST" for the first time, it takes a few seconds.)
4. Check the labels on the top to see if "Model's Guess" is the same as "Answer".

## *digit_recognition.py*

This program makes use of the trained model. It has a canvas for the user to write the digit on the screen. Then the model guesses which digit that the user has written. 

Instructions:
1. Click and drag  on the canvas to write the digit.
2. Click the button on the bottom-left that says "Predict".
3. The label at the bottom would tell you the model's guess, "Digit: X".
4. Click the button "Clear" to erase everything on the canvas.
