import cv2
import os
import onnxruntime as ort
from PIL import Image
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import sys


def preprocess_image(image_path):
    # خواندن تصویر با PIL و تبدیل به RGB
    pil_image = Image.open(image_path).convert("RGB")

    # ذخیره نسخه‌ی اصلی برای برگردوندن بعداً
    original_image = np.array(pil_image)

    # تبدیل به BGR برای سازگاری با مدل آموزش‌دیده با OpenCV
    bgr_image = original_image[..., ::-1]  # معکوس کردن کانال‌ها: RGB ➜ BGR

    # Resize
    resized_image = Image.fromarray(bgr_image).resize((512, 512))

    # نرمال‌سازی و تغییر محورها
    image = np.array(resized_image) / 255.0
    image = np.moveaxis(image, -1, 0)

    return np.expand_dims(image, axis=0).astype(np.float32), original_image


class WaterSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Water ")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        self.model = self.load_model()  # بارگذاری مدل ONNX

        self.style = ttk.Style("cyborg")
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=BOTH, expand=True)

        self.original_image = None
        self.segmented_image = None
        self.before_image = None
        self.after_image = None
        self.compare_button = None
        self.scan_image = None

        self.create_main_page()

    def load_model(self):
	    if getattr(sys, 'frozen', False):
	        base_path = sys._MEIPASS
	    else:
		# وقتی برنامه به صورت اسکریپت اجرا میشه
	        base_path = os.path.dirname(os.path.abspath(__file__))

	    model_path = os.path.join(base_path, "unet.onnx")
	    session = ort.InferenceSession(model_path)
	    return session

    def create_main_page(self):
        self.clear_frame()

        title = ttk.Label(self.main_frame, text="نرم افزار تشخیص و تحلیل منابع آبی", font=("B Nazanin", 20, "bold"))
        title.pack(pady=20)

        style = ttk.Style()
        style.configure("Big.TButton", font=("B Nazanin", 13), padding=30)

        ttk.Button(self.main_frame, text="Water Body Segmentation", style="Big.TButton", bootstyle=PRIMARY, width=30, command=self.open_scan_page).pack(pady=20)
        ttk.Button(self.main_frame, text="Analysis of Water Loss Over Time", style="Big.TButton", bootstyle=INFO, width=30, command=self.open_compare_page).pack(pady=20)
        ttk.Button(self.main_frame, text="Exit", style="Big.TButton", bootstyle=DANGER, width=30, command=self.root.quit).pack(pady=20)

    def open_scan_page(self):
        self.clear_frame()

        ttk.Label(self.main_frame, text="Water Body Segmentation", font=("Helvetica", 16)).pack(pady=10)
        ttk.Button(self.main_frame, text="Upload image", bootstyle=SECONDARY, command=self.upload_image).pack(pady=5)
        ttk.Button(self.main_frame, text="Back", bootstyle=WARNING, command=self.create_main_page).pack(pady=5)

    def open_compare_page(self):
        self.clear_frame()

        ttk.Label(self.main_frame, text="Analysis of Water Loss Over Time", font=("Helvetica", 16)).pack(pady=10)

        # دکمه بازگشت بالای بقیه دکمه‌ها
        ttk.Button(self.main_frame, text="Back", bootstyle=WARNING, command=self.create_main_page).pack(pady=5)


        # فریم افقی برای دکمه‌های آپلود و مقایسه
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)

        # دکمه‌ها به صورت افقی کنار هم
        ttk.Button(self.button_frame, text="‌Before image", bootstyle=SECONDARY, command=self.upload_before_image).pack(side=LEFT, padx=150)
        ttk.Button(self.button_frame, text="After image", bootstyle=SECONDARY, command=self.upload_after_image).pack(side=LEFT, padx=150)

        # دکمه مقایسه تفاوت به طور موقت ساخته نمی‌شه اینجا، چون توی display_compare_images ساخته می‌شه

        # فریم برای نمایش تصاویر مقایسه
        self.compare_frame = ttk.Frame(self.main_frame)
        self.compare_frame.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.original_image = str(file_path)
            ttk.Button(self.main_frame, text="Process Image!", bootstyle=SUCCESS, command=self.run_model).pack(pady=5)
            img = Image.open(self.original_image).resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.scan_image = ttk.Label(self.main_frame, image=img_tk)
            self.scan_image.image = img_tk
            self.scan_image.pack(padx=5)

    def upload_before_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.before_image = file_path
            self.display_compare_images()

    def upload_after_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.after_image = file_path
            self.display_compare_images()

    def display_compare_images(self):
        for widget in self.compare_frame.winfo_children():
            widget.destroy()

        # تصویر قبل (سمت چپ)
        if self.before_image:
            img = Image.open(self.before_image).resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            label = ttk.Label(self.compare_frame, image=img_tk)
            label.image = img_tk
            label.grid(row=0, column=0, padx=20)
        else:
            placeholder = ttk.Label(self.compare_frame, text=" ", width=40)
            placeholder.grid(row=0, column=0)

        # تصویر بعد (سمت راست)
        if self.after_image:
            img = Image.open(self.after_image).resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            label = ttk.Label(self.compare_frame, image=img_tk)
            label.image = img_tk
            label.grid(row=0, column=1, padx=20)
        else:
            placeholder = ttk.Label(self.compare_frame, text=" ", width=40)
            placeholder.grid(row=0, column=1)

        self.compare_frame.grid_columnconfigure(0, weight=1)
        self.compare_frame.grid_columnconfigure(1, weight=1)

        # دکمه زیر تصاویر
        if self.before_image and self.after_image:
            if self.compare_button:
                self.compare_button.destroy()
            self.compare_button = ttk.Button(
                self.compare_frame,
                text="Start Analysis!",
                bootstyle=SUCCESS,
                command=self.compare_masks
            )
            # قرار دادن دکمه در ردیف دوم و وسط
            self.compare_button.grid(row=1, column=0, columnspan=2, pady=10)

    def run_model(self):
        if self.original_image is None:
            return

        if self.scan_image:
            self.scan_image.destroy()

        input_tensor, original_image = preprocess_image(self.original_image)

        # استفاده از مدل ONNX برای پیش‌بینی
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        output = self.model.run([output_name], {input_name: input_tensor})[0]

        # تبدیل خروجی به ماسک
        mask = np.argmax(output, axis=1)[0]
        mask_color = np.zeros((512, 512, 3), dtype=np.uint8)
        mask_color[mask == 1] = [0, 0, 255]
        overlay = cv2.addWeighted(cv2.resize(original_image, (512, 512)), 0.5, mask_color, 0.5, 0)

        self.segmented_image = Image.fromarray(overlay).resize((300, 300))
        img_tk = ImageTk.PhotoImage(self.segmented_image)
        panel = ttk.Label(self.main_frame, image=img_tk)
        panel.image = img_tk
        panel.pack(pady=5)

    def compare_masks(self):
        if self.before_image is None or self.after_image is None:
            return

        if self.compare_button:
            self.compare_button.destroy()
            self.compare_button = None

        for widget in self.compare_frame.winfo_children():
            widget.destroy()

        before_tensor, before_orig = preprocess_image(self.before_image)
        after_tensor, after_orig = preprocess_image(self.after_image)

        # تبدیل به numpy و استفاده از onnx
        before_input = before_tensor
        after_input = after_tensor

        before_out = self.model.run(None, {"input": before_input})[0]
        after_out = self.model.run(None, {"input": after_input})[0]

        mask_before = np.argmax(before_out, axis=1)[0]
        mask_after = np.argmax(after_out, axis=1)[0]

        dry_area = (mask_before == 1) & (mask_after == 0)

        # اضافه‌کردن ماسک به تصویر قبل
        before_overlay = before_orig.copy()
        before_overlay = cv2.resize(before_overlay, (512, 512))
        mask_before_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        mask_before_rgb[mask_before == 1] = [0, 0, 255]
        before_overlay = cv2.addWeighted(before_overlay, 0.6, mask_before_rgb, 0.4, 0)

        # اضافه‌کردن نواحی خشک‌شده به تصویر بعد
        highlight = np.zeros((512, 512, 3), dtype=np.uint8)
        highlight[dry_area] = [255, 0, 0]
        after_resized = cv2.resize(after_orig, (512, 512))
        mask_after_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        mask_after_rgb[mask_after == 1] = [0, 0, 255]
        mask_after_rgb[dry_area] = [255, 0, 0]
        overlay = cv2.addWeighted(after_resized, 0.6, mask_after_rgb, 0.5, 0)

        before_img = Image.fromarray(before_overlay).resize((300, 300))
        before_tk = ImageTk.PhotoImage(before_img)
        before_panel = ttk.Label(self.compare_frame, image=before_tk)
        before_panel.image = before_tk
        before_panel.pack(side=LEFT, padx=5)

        after_img = Image.fromarray(overlay).resize((300, 300))
        after_tk = ImageTk.PhotoImage(after_img)
        after_panel = ttk.Label(self.compare_frame, image=after_tk)
        after_panel.image = after_tk
        after_panel.pack(side=LEFT, padx=5)

        area1 = np.sum(mask_before == 1)
        area2 = np.sum(mask_after == 1)

        if area1 == 0:
            change_percent = 0
        else:
            change_percent = ((area1 - area2) / area1) * 100

        label = ttk.Label(self.main_frame, text=f"Water Loss Percentage : {change_percent:.2f}٪")
        label.pack(pady=10)

        self.before_image = None
        self.after_image = None



    def clear_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = ttk.Window(themename="cyborg")
    app = WaterSegmentationApp(root)
    root.mainloop()
