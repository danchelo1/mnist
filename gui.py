import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow import keras
import os

# Загрузить модели
m1 = keras.models.load_model("models/model_mnist.h5")
m2 = keras.models.load_model("models/model_augmented.h5")


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Распознавание рукописных цифр — сравнение двух сетей")
        self.canvas_size = 280
        self.image_size = 28

        # холст tkinter
        self.canvas = tk.Canvas(master, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=4, pady=10, padx=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw_pil = ImageDraw.Draw(self.image)

        clear_btn = tk.Button(master, text="Очистить", command=self.clear)
        clear_btn.grid(row=1, column=0, pady=5)

        predict_btn = tk.Button(master, text="Predict (now)", command=self.predict)
        predict_btn.grid(row=1, column=1, pady=5)

        save_btn = tk.Button(master, text="Сохранить изображение", command=self.save_image)
        save_btn.grid(row=1, column=2, pady=5)

        quit_btn = tk.Button(master, text="Выход", command=master.quit)
        quit_btn.grid(row=1, column=3, pady=5)

        self.label_m1 = tk.Label(master, text="Model 1: —", font=("Arial", 14))
        self.label_m1.grid(row=2, column=0, columnspan=2, sticky='w', padx=10)
        self.label_m2 = tk.Label(master, text="Model 2: —", font=("Arial", 14))
        self.label_m2.grid(row=2, column=2, columnspan=2, sticky='w', padx=10)

        self.last_x, self.last_y = None, None
        self.line_width = 18

    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x is None:
            self.last_x, self.last_y = x, y

        self.canvas.create_line(self.last_x, self.last_y, x, y, width=self.line_width, fill='black', capstyle=tk.ROUND,
                                smooth=True)
        self.draw_pil.line([self.last_x, self.last_y, x, y], fill=0, width=self.line_width)
        self.last_x, self.last_y = x, y

    def on_release(self, event):
        self.last_x, self.last_y = None, None
        self.predict()

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw_pil = ImageDraw.Draw(self.image)
        self.label_m1.config(text="Model 1: —")
        self.label_m2.config(text="Model 2: —")

    def save_image(self):
        self.image.save("drawn_digit.png")
        print("Saved to drawn_digit.png")

    def preprocess(self):
        img = self.image.copy()
        img = ImageOps.invert(img)
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        arr = np.array(img).astype('float32') / 255.0
        arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self):
        x = self.preprocess()
        p1 = m1.predict(x)[0]
        p2 = m2.predict(x)[0]
        pred1 = int(np.argmax(p1))
        pred2 = int(np.argmax(p2))
        conf1 = float(p1[pred1])
        conf2 = float(p2[pred2])
        self.label_m1.config(text=f"Model 1: {pred1} (conf {conf1:.3f})")
        self.label_m2.config(text=f"Model 2: {pred2} (conf {conf2:.3f})")
        print("Model1 probs:", np.round(p1, 3))
        print("Model2 probs:", np.round(p2, 3))


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
