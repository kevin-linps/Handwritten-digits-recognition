from sklearn.datasets import load_digits
from joblib import load
import numpy as np
import tensorflow as tf
from random import randint
from tkinter import*
from tkinter.ttk import*

class DigitVisualizer(object):

    def __init__(self):

        self.root = Tk()

        self.options = options = ["", "scikit-learn", "MNIST"]
        self.database = StringVar(self.root)
        self.database.set(self.options[0])
        
        self.Fr = Frame(self.root)
        self.L1 = Label(self.Fr, text="Data set:")
        self.L2 = Label(self.Fr, text="Answer:", width=10)
        self.L3 = Label(self.Fr, text="Model's Guess:", width=15)
        self.Op = OptionMenu(self.Fr, self.database, *self.options)
        self.B1 = Button(self.Fr, text="Choose an image", command=self.choose_image)
        self.Ca = Canvas(self.root, width=450, height=450, bg="white")

        self.pixels = []

        self.setup()

    def setup(self):

        # Construct the screeen
        self.root.resizable(width=False, height=False)
        self.root.geometry("500x500")
        self.root.title("Handwritten Digits Visualizer")

        self.Op.config(width=10)

        # Place the widgets on the screen
        self.L1.pack(side=LEFT)
        self.Op.pack(side=LEFT)
        self.L2.pack(side=LEFT, padx=5)
        self.L3.pack(side=LEFT, padx=5)
        self.B1.pack(side=LEFT, padx=5)
        self.Fr.pack(pady=3)
        self.Ca.pack(pady=3)

        self.root.mainloop()

    def choose_image(self):
        
        if self.database.get() == "scikit-learn":

            n = randint(0, 1797)
            digit = self.predict_digit(n)
            self.L2.config(text = "Answer: " + str(sk_y[n]))
            self.L3.config(text = "Model's Guess: " + str(digit))
            self.paint_digit(n)

        elif self.database.get() == "MNIST":

            n = randint(0, 70000)
            digit = self.predict_digit(n)
            self.L2.config(text = "Answer: " + str(tf_y[n]))
            self.L3.config(text = "Model's Guess: " + str(digit))
            self.paint_digit(n)

    def predict_digit(self, n):

        if self.database.get() == "scikit-learn":
            data = sk_x[n].flatten()
            return sk_model.predict([data])[0]
        elif self.database.get() == "MNIST":
            data = tf_x[n].reshape(1, 28, 28, 1)/255
            respond = tf_model.predict(data)
            return np.argmax(respond)

    def paint_digit(self, n):

        for item in self.pixels:
            self.Ca.delete(item)
        self.pixels = []

        if self.database.get() == "scikit-learn":
            grid = sk_x[n]
            L = 450/8
        elif self.database.get() == "MNIST":
            grid = tf_x[n]
            L = 450/28

        for h in range(len(grid)):
            for w in range(len(grid[0])):
                if grid[h][w] != 0:
                    colour = self.create_colour_code(grid[h][w])
                    rect = self.Ca.create_rectangle(w*L, h*L, (w+1)*L, (h+1)*L, 
                            fill = colour, outline = colour)
                    self.pixels.append(rect) 

    def create_colour_code(self, number):

        if self.database.get() == "scikit-learn": 
            number *= 16

        H = hex(abs(255-number))[2:]
        if (len(H) == 1):
            H = "0" + H
        return "#" + H * 3

if __name__ == "__main__":

    # Load models externally so they are loaded when the screen is up
    sk_model = load('svm_model.joblib')
    tf_model = tf_model = tf.keras.models.load_model('CNN_model.h5')

    # Load the digits database from scikit-learn
    sk_digits = load_digits()
    sk_x = sk_digits.data.reshape(1797, 8, 8).astype('int')
    sk_y = sk_digits.target

    # Load MNIST database using TensorFlow
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    tf_x = np.concatenate((X_train, X_test))
    tf_y = np.concatenate((y_train, y_test))

    DigitVisualizer()