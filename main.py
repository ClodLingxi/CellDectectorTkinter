import tkinter as tk
from tkinter import filedialog, messagebox, Menu, Toplevel
from tkinter import ttk

from PIL import Image, ImageTk, ImageOps, ImageFilter
import os
import math
import cv2
import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class ImageLoaderApp:
    def __init__(self, root):
        self.image_id = None
        self.root = root
        self.root.title("Image Loader")
        self.root.resizable(False, False)

        self.menu = None
        self.create_menu()
        self.file_menu = self.create_load_file_menu()
        self.model_menu = self.create_model_detect_menu()
        self.tools_menu = self.create_tools_menu()

        self.label = tk.Label(root, text="Select an image or a folder to load images")
        self.label.pack(pady=10)

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
        self.processed_image: Image.ImageFile = None
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

        self.frame = tk.Frame(root)
        self.frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame, bg='gray')
        self.canvas.grid(row=0, column=0, sticky='nsew')

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        self.scroll_x = tk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scroll_y = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        self.scroll_x.grid(row=1, column=0, sticky='ew')
        self.scroll_y.grid(row=0, column=1, sticky='ns')

    def test(self):
        self.load_image()
        self.load_model()
        self.detect_objects()

    def create_menu(self):
        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)

    def create_load_file_menu(self):
        file_menu = Menu(self.menu, tearoff=1)
        self.menu.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Load Folder", command=self.load_folder)
        file_menu.add_command(label="Save Image", command=self.save_image)
        file_menu.add_command(label="Test", command=self.test)

        return file_menu

    def create_model_detect_menu(self):
        detect_menu = Menu(self.menu, tearoff=1)
        self.menu.add_cascade(label="Detect", menu=detect_menu)

        detect_menu.add_command(label="Load Model(pt)", command=self.load_model)
        detect_menu.add_command(label="Start Detect", command=self.detect_objects)

        return detect_menu

    def create_tools_menu(self):
        tools_menu = Menu(self.menu, tearoff=1)
        self.menu.add_cascade(label="Tools", menu=tools_menu)

        tools_menu.add_command(label="Enable Ruler", command=self.toggle_ruler)
        tools_menu.add_command(label="Enable Selector", command=self.toggle_selector)

        return tools_menu

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
        self.tools_menu.entryconfig(1, label="Disable Ruler" if self.ruler_enabled else "Enable Ruler")
        self.tools_menu.entryconfig(2, label="Enable Selector")
        self.clear_ruler()

    def toggle_selector(self):
        self.selector_enabled = not self.selector_enabled
        self.ruler_enabled = False
        self.tools_menu.entryconfig(2, label="Disable Selector" if self.selector_enabled else "Enable Selector")
        self.tools_menu.entryconfig(1, label="Enable Ruler")
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
            self.ruler_line = self.canvas.create_line(x1, y1, x2, y2, fill='blue')
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
            self.canvas.delete(self.image_id)

            img = self.processed_image.copy()
            new_size = (int(self.processed_image.width * self.scale), int(self.processed_image.height * self.scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            self.image = ImageTk.PhotoImage(img)

            self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            self.clear_ruler()

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if file_path:
            self.model = file_path
            model_name = os.path.basename(file_path)
            self.model_name_label.config(text=f"Model Loaded: {model_name}")
            messagebox.showinfo("Model Loaded", f"Model loaded from {file_path}")

    def detect_objects(self, output_name='output.jpg'):
        if self.processed_image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        if self.model is None:
            messagebox.showerror("Error", "No model loaded.")
            return

        self.processed_image.save(output_name)

        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=self.model,
            confidence_threshold=0.3,
            device="cuda:0",  # or 'cuda:0'
        )

        result = get_sliced_prediction(output_name,
                                       detection_model,
                                       slice_height=640,
                                       slice_width=640,
                                       overlap_height_ratio=0.2,
                                       overlap_width_ratio=0.2
                                       )
        result.export_visuals(export_dir="demo_data/")
        self.processed_image = Image.open("demo_data/prediction_visual")

        # results = self.model(output_path)
        # for result in results:
        #     result.save(output_path)
        #

        self.clear_ruler()
        self.clear_selection()
        self.update_image()

    def apply_selection(self, x1, y1, x2, y2, temp_file='cropped_image.jpg'):
        if self.processed_image:
            left = min(int(x1 / self.scale), int(x2 / self.scale))
            top = min(int(y1 / self.scale), int(y2 / self.scale))
            right = max(int(x1 / self.scale), int(x2 / self.scale))
            bottom = max(int(y1 / self.scale), int(y2 / self.scale))

            cropped_image = self.processed_image.crop((left, top, right, bottom))
            cropped_image.save(temp_file)

            editor_window = Toplevel(self.root)
            editor_window.title("Image Editor")
            editor = ImageEditor(editor_window, temp_file)

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


class ImageEditor:
    def __init__(self, root, file_name):
        self.root = root
        self.root.title("Image Adjuster")

        self.image_label = tk.Label(root)
        self.image_label.pack()
        self.tk_image = None

        self.radius_label = tk.Label(root)
        self.radius_label.pack()

        self.threshold1_scale, self.threshold1_scale_entry = self.create_scale_entry(root, "Threshold1",
                                                                                     0, 200, 50, 1,
                                                                                     self.update_image)

        self.threshold2_scale, self.threshold2_scale_entry = self.create_scale_entry(root, "Threshold2",
                                                                                     0, 200, 30, 1,
                                                                                     self.update_image)

        self.threshold3_scale, self.threshold3_scale_entry = self.create_scale_entry(root, "Threshold2",
                                                                                     1, 300, 1, 1,
                                                                                     self.update_image)

        self.data = (0, 0, 0)
        self.file_name = file_name
        self.update_image()

    def create_scale_entry(self, root, label, from_, to, initial, resolution, command):
        frame = tk.Frame(root)
        frame.pack()

        scale = tk.Scale(frame, from_=from_, to=to, resolution=resolution, orient=tk.HORIZONTAL, label=label,
                         command=lambda v: self.on_scale_change(v, scale, entry, command))
        scale.set(initial)
        scale.pack(side=tk.LEFT)

        entry = tk.Entry(frame, width=5)
        entry.insert(0, str(initial))
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", lambda event: self.on_entry_change(scale, entry, command))

        return scale, entry

    def on_scale_change(self, value, scale, entry, command):
        entry.delete(0, tk.END)
        entry.insert(0, str(value))
        command()

    def on_entry_change(self, scale, entry, command):
        try:
            value = float(entry.get())
            scale.set(value)
            command()
        except ValueError:
            pass

    def get_distance(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def cell_membrane(self, file_name, threshold1=50, threshold2=30, threshold3=160):
        output_filename = 'output.jpg'
        image = cv2.imread(file_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        min_shape = min(image.shape[0], image.shape[1])
        max_shape = min(image.shape[0], image.shape[1])
        cell_center = (int(image.shape[0] / 2), int(image.shape[1] / 2))
        result = {'image': image, 'center': cell_center, 'radius': -1}

        im = image.copy()
        del image

        circle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                                  minDist=min_shape / 3, param1=threshold1, param2=threshold2,
                                  minRadius=int(min_shape / 3), maxRadius=0)
        radius = 0
        (x, y) = (0, 0)

        if circle is not None:
            for a, b, c in circle[0]:
                if c > radius:
                    print(radius)
                    (x, y) = (a, b)
                    radius = c

                center = (int(x), int(y))
                radius = int(radius)

                cv2.circle(im, center, radius, (0, 255, 0), 2)

                result = {'image': im, 'center': center, 'radius': radius}

        cv2.imwrite(output_filename, im)
        del im

        image = cv2.imread(output_filename, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, threshold3, 255, cv2.THRESH_BINARY)

        result_count = 0
        inner_radius = 0
        for _radius in range(int(result['radius'] / 3), result['radius']):

            mask = np.zeros_like(binary_image, dtype=np.uint8)

            cv2.circle(mask, result['center'], _radius, 255, -1)

            circle_area = cv2.bitwise_and(binary_image, binary_image, mask=mask)
            count = np.sum(circle_area == 0)

            if _radius > 2 and result_count > count:
                inner_radius = _radius
                break
            result_count = count

        image = cv2.imread(file_name)

        if result['radius'] == -1:
            print('No Circle')
            return output_filename, inner_radius, 0

        cv2.circle(image, result['center'], inner_radius, (0, 0, 255), 1)
        cv2.circle(image, result['center'], result['radius'], (0, 255, 0), 1)
        cv2.imwrite(output_filename, image)

        return output_filename, inner_radius, result['radius']

    def update_image(self, event=None):
        if self.file_name:
            threshold1 = self.threshold1_scale.get()
            threshold2 = self.threshold2_scale.get()
            threshold3 = self.threshold3_scale.get()

            if (threshold1, threshold2, threshold3) == self.data:
                return

            self.data = (threshold1, threshold2, threshold3)

            output, inner_radius, outer_radius = self.cell_membrane(self.file_name, threshold1, threshold2, threshold3)

            final_image = Image.open(output)
            self.tk_image = ImageTk.PhotoImage(final_image)
            self.image_label.config(image=self.tk_image)
            self.radius_label.config(text=str(abs(outer_radius - inner_radius)))


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLoaderApp(root)
    root.mainloop()
