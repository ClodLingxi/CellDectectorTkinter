import tkinter as tk
from tkinter import filedialog, messagebox, Menu
from PIL import Image, ImageTk, ImageOps, ImageFilter
import os
import math
from ultralytics import YOLO


class ImageLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Loader")
        self.root.resizable(False, False)

        self.create_menu()

        self.label = tk.Label(root, text="Select an image or a folder to load images")
        self.label.pack(pady=10)

        self.load_model_button = tk.Button(root, text="Load Model (.pt)", command=self.load_model)
        self.load_model_button.pack(pady=10)

        self.detect_button = tk.Button(root, text="Detect", command=self.detect_objects)
        self.detect_button.pack(pady=10)

        self.ruler_button = tk.Button(root, text="Enable Ruler", command=self.toggle_ruler)
        self.ruler_button.pack(pady=10)

        self.selector_button = tk.Button(root, text="Enable Selector", command=self.toggle_selector)
        self.selector_button.pack(pady=10)

        self.save_button = tk.Button(root, text="Save Image", command=self.save_image)
        self.save_button.pack(pady=10)

        self.canvas = tk.Canvas(root, width=800, height=600, bg='gray')
        self.canvas.pack()

        self.prev_button = tk.Button(root, text="<< Prev", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(root, text="Next >>", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT, padx=10)

        self.image_size_label = tk.Label(root, text="")
        self.image_size_label.pack(pady=10)

        self.model_name_label = tk.Label(root, text="")
        self.model_name_label.pack(pady=10)

        self.unit_label = tk.Label(root, text="Unit: pixels")
        self.unit_label.pack(pady=10)

        self.image_files = []
        self.current_image_index = -1
        self.image = None
        self.original_image = None
        self.processed_image = None
        self.ruler_enabled = False
        self.selector_enabled = False
        self.ruler_start = None
        self.ruler_end = None
        self.ruler_line = None
        self.ruler_text = None
        self.scale = 1.0
        self.model = None
        self.selection_rectangle = None
        self.selection_start = None
        self.unit_scale = 1.0

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        self.root.bind("<MouseWheel>", self.on_mouse_wheel)

    def create_menu(self):
        menu = Menu(self.root)
        self.root.config(menu=menu)

        file_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Load Folder", command=self.load_folder)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            self.image_files = [file_path]
            self.current_image_index = 0
            self.show_image()

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            self.image_files.sort()
            self.current_image_index = 0
            self.show_image()

    def show_image(self):
        if self.image_files and 0 <= self.current_image_index < len(self.image_files):
            img_path = self.image_files[self.current_image_index]
            self.original_image = Image.open(img_path)
            self.processed_image = self.original_image.copy()
            self.scale = 1.0
            self.update_image()
            self.root.title(f"Image Loader - {os.path.basename(img_path)}")
            self.clear_ruler()
            self.image_size_label.config(
                text=f"Image Size: {self.original_image.size[0]} x {self.original_image.size[1]}")
            self.scale_image_to_fit()

    def scale_image_to_fit(self):
        if self.processed_image:
            img_width, img_height = self.processed_image.size
            canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
            self.scale = min(canvas_width / img_width, canvas_height / img_height)
            self.update_image()

    def next_image(self):
        if self.image_files:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            self.show_image()

    def prev_image(self):
        if self.image_files:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
            self.show_image()

    def toggle_ruler(self):
        self.ruler_enabled = not self.ruler_enabled
        self.selector_enabled = False
        self.ruler_button.config(text="Disable Ruler" if self.ruler_enabled else "Enable Ruler")
        self.selector_button.config(text="Enable Selector")
        self.clear_ruler()

    def toggle_selector(self):
        self.selector_enabled = not self.selector_enabled
        self.ruler_enabled = False
        self.selector_button.config(text="Disable Selector" if self.selector_enabled else "Enable Selector")
        self.ruler_button.config(text="Enable Ruler")
        self.clear_selection()

    def on_canvas_click(self, event):
        if self.ruler_enabled:
            if self.ruler_start is None:
                self.ruler_start = (event.x, event.y)
            else:
                self.ruler_end = (event.x, event.y)
                self.calculate_unit_scale()
                self.ruler_start = None
                self.ruler_end = None
        elif self.selector_enabled:
            self.selection_start = (event.x, event.y)
            if self.selection_rectangle:
                self.canvas.delete(self.selection_rectangle)
            self.selection_rectangle = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='blue')

    def on_mouse_move(self, event):
        if self.ruler_enabled and self.ruler_start is not None:
            if self.ruler_line:
                self.canvas.delete(self.ruler_line)
            if self.ruler_text:
                self.canvas.delete(self.ruler_text)

            x1, y1 = self.ruler_start
            x2, y2 = event.x, event.y
            self.ruler_line = self.canvas.create_line(x1, y1, x2, y2, fill='red')
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / self.scale
            self.ruler_text = self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=f"{distance:.2f} pixels",
                                                      fill='red')
        elif self.selector_enabled and self.selection_start is not None:
            x1, y1 = self.selection_start
            x2, y2 = event.x, event.y
            self.canvas.coords(self.selection_rectangle, x1, y1, x2, y2)

    def on_mouse_release(self, event):
        if self.selector_enabled and self.selection_start is not None:
            x1, y1 = self.selection_start
            x2, y2 = event.x, event.y
            self.apply_selection(x1, y1, x2, y2)
            self.selection_start = None

    def clear_ruler(self):
        if self.ruler_line:
            self.canvas.delete(self.ruler_line)
            self.ruler_line = None
        if self.ruler_text:
            self.canvas.delete(self.ruler_text)
            self.ruler_text = None
        self.ruler_start = None
        self.ruler_end = None

    def clear_selection(self):
        if self.selection_rectangle:
            self.canvas.delete(self.selection_rectangle)
            self.selection_rectangle = None
        self.selection_start = None

    def on_mouse_wheel(self, event):
        if self.processed_image:
            if event.delta > 0:
                self.scale *= 1.1
            else:
                self.scale *= 0.9
            self.update_image()

    def update_image(self):
        if self.processed_image:
            img = self.processed_image.copy()
            new_size = (int(self.processed_image.width * self.scale), int(self.processed_image.height * self.scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            self.image = ImageTk.PhotoImage(img)
            self.canvas.config(width=new_size[0], height=new_size[1])
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.clear_ruler()

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if file_path:
            self.model = YOLO(file_path)
            model_name = os.path.basename(file_path)
            self.model_name_label.config(text=f"Model Loaded: {model_name}")
            messagebox.showinfo("Model Loaded", f"Model loaded from {file_path}")

    def detect_objects(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        if self.model is None:
            messagebox.showerror("Error", "No model loaded.")
            return

        # Save the current image to a temporary file
        img_path = "temp_image.jpg"
        self.processed_image.save(img_path)

        # Perform detection
        results = self.model(img_path)

        # Display results on the canvas
        self.display_results(results)

    def display_results(self, results):
        # Clear the canvas
        self.canvas.delete("all")

        # Draw the image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # Draw the bounding boxes and labels
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                self.canvas.create_rectangle(x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale,
                                             outline='red', width=2)
                self.canvas.create_text(x1 * self.scale, y1 * self.scale, anchor=tk.NW, text=box.cls_name, fill='red')

        self.clear_ruler()

    def apply_selection(self, x1, y1, x2, y2):
        if self.processed_image:
            left = min(int(x1 / self.scale), int(x2 / self.scale))
            top = min(int(y1 / self.scale), int(y2 / self.scale))
            right = max(int(x1 / self.scale), int(x2 / self.scale))
            bottom = max(int(y1 / self.scale), int(y2 / self.scale))

            cropped_image = self.processed_image.crop((left, top, right, bottom))
            edge_image = cropped_image.filter(ImageFilter.FIND_EDGES)
            self.processed_image.paste(edge_image, (left, top))
            self.update_image()

    def save_image(self):
        if self.processed_image:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"),
                                                                ("All Files", "*.*")])
            if save_path:
                self.processed_image.save(save_path)
                messagebox.showinfo("Image Saved", f"Image saved to {save_path}")

    def calculate_unit_scale(self):
        if self.ruler_start and self.ruler_end:
            x1, y1 = self.ruler_start
            x2, y2 = self.ruler_end
            pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            unit = tk.simpledialog.askfloat("Set Unit", "Enter the real-world distance for the selected pixels:")
            if unit:
                self.unit_scale = pixel_distance / unit
                self.unit_label.config(text=f"Unit: {self.unit_scale:.2f} pixels per unit")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLoaderApp(root)
    root.mainloop()
