import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk, ImageEnhance, ImageOps
import cv2
import numpy as np

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Window")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.edit_button = tk.Button(root, text="Edit Image", command=self.open_editor)
        self.edit_button.pack()

        self.image = None
        self.tk_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.update_image()

    def update_image(self, image=None):
        if image:
            self.image = image
        if self.image:
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.tk_image)

    def open_editor(self):
        if self.image:
            editor_window = Toplevel(self.root)
            editor_window.title("Image Editor")
            editor = ImageEditorWindow(editor_window, self.image, self.update_image)

class ImageEditorWindow:
    def __init__(self, root, image, update_callback):
        self.root = root
        self.image = image
        self.update_callback = update_callback

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.brightness_scale = tk.Scale(root, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="Brightness", command=self.update_image)
        self.brightness_scale.set(1.0)
        self.brightness_scale.pack()

        self.contrast_scale = tk.Scale(root, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="Contrast", command=self.update_image)
        self.contrast_scale.set(1.0)
        self.contrast_scale.pack()

        self.horizontal_shift_scale = tk.Scale(root, from_=-100, to=100, resolution=1, orient=tk.HORIZONTAL, label="Horizontal Shift", command=self.update_image)
        self.horizontal_shift_scale.set(0)
        self.horizontal_shift_scale.pack()

        self.vertical_shift_scale = tk.Scale(root, from_=-100, to=100, resolution=1, orient=tk.HORIZONTAL, label="Vertical Shift", command=self.update_image)
        self.vertical_shift_scale.set(0)
        self.vertical_shift_scale.pack()

        self.detect_circles_button = tk.Button(root, text="Detect Circles", command=self.detect_circles)
        self.detect_circles_button.pack()

        self.save_button = tk.Button(root, text="Save Changes", command=self.save_changes)
        self.save_button.pack()

        self.tk_image = None
        self.update_image()

    def update_image(self, event=None):
        if self.image:
            brightness = self.brightness_scale.get()
            contrast = self.contrast_scale.get()
            h_shift = self.horizontal_shift_scale.get()
            v_shift = self.vertical_shift_scale.get()

            enhancer = ImageEnhance.Brightness(self.image)
            bright_image = enhancer.enhance(brightness)

            enhancer = ImageEnhance.Contrast(bright_image)
            final_image = enhancer.enhance(contrast)

            width, height = final_image.size
            final_image = ImageOps.expand(final_image, border=(100, 100, 100, 100), fill=(0, 0, 0))
            final_image = final_image.crop((100 - h_shift, 100 - v_shift, 100 + width - h_shift, 100 + height - v_shift))

            self.tk_image = ImageTk.PhotoImage(final_image)
            self.image_label.config(image=self.tk_image)

    def detect_circles(self):
        # Convert PIL image to OpenCV image
        open_cv_image = np.array(self.image.convert('RGB'))
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=0,
                                   maxRadius=0)

        print(circles)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(open_cv_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(open_cv_image, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Convert back to PIL image and update
        final_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(final_image)
        self.update_image(final_image)

    def save_changes(self):
        self.update_callback(self.image)


root = tk.Tk()
app = ImageEditor(root)
root.mainloop()
