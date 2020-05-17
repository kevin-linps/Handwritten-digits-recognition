"""
PART ONE: Importing data sets and models

This program uses some non-essential libraries in Python 3, particularly sklearn, joblib, numpy and
tensorflow. If these packages are not installed in your system, an error message would be printed
and the program quits.

"""
try:
    # Import the data sets and models
    from sklearn import datasets
    from joblib import load
    sklearn_digits = datasets.load_digits()
    sklearnX = sklearn_digits.data.reshape(1797, 8, 8).astype('int32')
    sklearnY = sklearn_digits.target
    sklearn_model = load('svm_model.joblib')

    import numpy as np
    import tensorflow as tf
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = X_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
    x_test  = X_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
    tf_model = tf.keras.models.load_model('CNN_model.h5')

except:
    print("Error occurred when fetching data sets or importing models.")
    quit()




"""
PART TWO: GUI Display

This part uses the essential libraries in Python to build the GUI. If you have Python 3 installed, 
the code from this point should execute no problem.

"""
# Import packages used for GUI development
from random import randint
from tkinter import*
from tkinter.ttk import*

# Create the window of the GUI
root = Tk()
root.resizable(width= False, height = False)
root.geometry("500x500")
root.title("Handwritten Digits Recognition")

# Set up the toolbar and the canvas
SIZE = 400
frame = Frame(root)
c = Canvas(root, width = SIZE, height = SIZE, bg = "gray")
frame.pack(pady = 3)
c.pack(pady = 3)

pixels = []
options = ["", "scikit-learn", "MNIST"]
selected_dataset = StringVar(root)
selected_dataset.set(options[0])

def create_colour_code(number, MNIST):

    if not MNIST: 
        number *= 16

    H = hex(abs(255-number))[2:]
    if (len(H) == 1):
        H = "0" + H
    return "#" + H * 3

def choose_image():
    global pixels, L2, L3

    # Remove previous image
    for item in pixels:
        c.delete(item)
    pixels = []

    isMNIST = True

    # Choose a random image from the given data set and change labels
    if selected_dataset.get() == "scikit-learn":

        i = randint(0, 1797)
        grid = sklearnX[i]
        L = SIZE / len(grid[0])
        L2.config(text = "Answer: " + str(sklearnY[i]))
        L3.config(text = "Predict: " + str(sklearn_model.predict([sklearn_digits.data[i]])[0]))

        isMNIST = False

    elif selected_dataset.get() == "MNIST":
        i = randint(0, 70000)

        if i < 60000:

            # Use tf_model to predict the written digit
            n = x_train[i].reshape(1, 28, 28, 1)
            respond = tf_model.predict(n)
            prediction = str(np.argmax(respond))

            grid = X_train[i]
            answer = str(y_train[i])
        else:

            i = i - 60000

            # Use tf_model to predict the written digit
            n = x_test[i].reshape(1, 28, 28, 1)
            respond = tf_model.predict(n)
            prediction = str(np.argmax(respond))

            grid = X_test[i]
            answer = str(y_test[i])

        # Change the labels on the GUI
        L2.config(text = "Answer: " + answer)
        L3.config(text = "Predict: " + prediction)

        L = SIZE / len(grid[0])

    else:
        return False

    # Paint the image onto the canvas
    for h in range(len(grid)):
        for w in range(len(grid[0])):
            colour = create_colour_code(grid[h][w], isMNIST)
            rect = c.create_rectangle(w*L, h*L, (w+1)*L, (h+1)*L, fill = colour, outline = colour)
            pixels.append(rect)
    
    return True

# Declare the widgets
L1 = Label(frame, text = "Data set:")
L2 = Label(frame, text = "Answer:")
L3 = Label(frame, text = "Predict:")
Op = OptionMenu(frame, selected_dataset, *options)
B1 = Button(frame, text = "Choose a random image", command = choose_image)

# Configure the widgets
L2.config(width = 10)
L3.config(width = 10)
Op.config(width = 10)

# Place the widgets on the screen
L1.pack(side = LEFT)
Op.pack(side = LEFT)
L2.pack(side = LEFT, padx = 5)
L3.pack(side = LEFT, padx = 5)
B1.pack(side = LEFT, padx = 5)

root.mainloop()