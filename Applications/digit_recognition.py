import numpy as np
import tensorflow as tf
from tkinter import*
from tkinter.ttk import*

class DigitRecognition(object):

    def __init__(self):

        self.root = Tk()

        self.pw = 490/28

        self.C = Canvas(self.root, bg='gray', width=490, height=490)

        self.F = Frame(self.root)
        self.B1 = Button(self.F, text="Predict", command = self.predict_digit)
        self.B2 = Button(self.F, text="Clear", command = self.clear_canvas)
        self.L = Label(self.F, text="Digit: Unknown")

        self.colours = np.zeros(shape=(28, 28)).astype('int')
        self.pixels = np.array([[self.create_rect(x, y) for x in range(28)] for y in range(28)])

        self.setup()    # configure the screen and the locations of the widgets

    def setup(self):

        self.B1.pack(side=LEFT, padx=5)
        self.B2.pack(side=LEFT, padx=5)
        self.L.pack(side=LEFT, padx=5)

        self.C.grid(row=0, column=0)
        self.F.grid(row=1, column=0, pady=10, sticky=W)

        self.root.resizable(width=False, height=False)
        self.root.geometry("495x550+300+0")
        self.root.title("Handwritten Digits Recognition")

        self.C.bind("<B1-Motion>", self.draw)

        self.root.mainloop()

    def create_colour_code(self, number):

        # convert to hex RCB colour code
        H = hex(number)[2:]
        if (len(H) == 1):
            H = "0" + H
        return "#" + H * 3

    def create_rect(self, x, y):
        colour = self.create_colour_code(self.colours[x][y])
        return self.C.create_rectangle(x*self.pw, y*self.pw, (x+1)*self.pw, (y+1)*self.pw, 
                                       fill = colour, outline = colour)

    def update_colour(self, x, y, value):
        if value > self.colours[y,x]:
            self.colours[y,x] = value
            self.C.itemconfig(self.pixels[y,x], fill = self.create_colour_code(value))

    def draw(self, event):
        
        # check if cursor is on the canvas
        if 0 < event.x < 490 and 0 < event.y < 490:

            # update which pixel it is currently on
            X = int(event.x/self.pw)
            Y = int(event.y/self.pw)

            # update the colours of the pixelss
            self.update_colour(X, Y, 255)
            if X-1 >= 0 and Y-1 >= 0:
                self.update_colour(X-1, Y-1, 170)
                self.update_colour(X-1, Y, 170)
                self.update_colour(X, Y-1, 170)

    def predict_digit(self):

        # change the shape of the array so it can be fed into CNN
        x = np.array(self.colours).reshape((1, 28, 28, 1))/255

        # determine the digit written and update the label
        respond = model.predict(x)
        if np.argmax(respond) > 0.95:
            prediction = str(np.argmax(respond))
            self.L.config(text="Digit: "+prediction)
        else:
            self.L.config(text="Digit: Unknown")

    def clear_canvas(self):

        for rectangle in self.pixels:
            self.C.delete(rectangle)

        self.colours = np.zeros(shape=(28, 28)).astype('int')
        self.pixels = np.array([[self.create_rect(x, y) for x in range(28)] for y in range(28)])
        self.L.config(text="Digit: N/A")

if __name__ == "__main__":
    model = tf.keras.models.load_model('CNN_model.h5')
    DigitRecognition()