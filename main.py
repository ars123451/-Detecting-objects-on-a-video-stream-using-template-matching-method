import cv2
import tkinter as tk
from PIL import Image, ImageTk


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection with Template Matching")

        self.capture = cv2.VideoCapture(0)
        _, self.frame = self.capture.read()
        self.template = None
        self.selecting_template = False
        self.selection_start = None
        self.selection_end = None

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.btn_create_template = tk.Button(root, text="Create Template", command=self.start_template_selection)
        self.btn_create_template.pack()

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        self.update()

    def start_template_selection(self):
        self.selecting_template = True
        self.template = None

    def on_mouse_press(self, event):
        if self.selecting_template:
            self.selection_start = (event.x, event.y)

    def on_mouse_drag(self, event):
        if self.selecting_template and self.selection_start:
            self.selection_end = (event.x, event.y)
            self.display_frame()

    def on_mouse_release(self, event):
        if self.selecting_template and self.selection_start and self.selection_end:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            self.template = self.frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            self.selecting_template = False
            self.selection_start = None
            self.selection_end = None

    def update(self):
        _, self.frame = self.capture.read()
        if self.template is not None:
            position = self.detect_object()
            self.display_frame(position)
        else:
            self.display_frame()
        self.root.after(10, self.update)

    def display_frame(self, position=None):
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        if self.selection_start and self.selection_end:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.photo = photo

        if position is not None:
            print(f"Object Detected in {position} Part")

    def detect_object(self):
        result = cv2.matchTemplate(self.frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        x, y = max_loc
        w, h = self.template.shape[1], self.template.shape[0]

        cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        height, width = self.frame.shape[:2]
        part_width = width // 2
        part_height = height // 2

        if x < part_width:
            if y < part_height:
                return "Top Left"
            else:
                return "Bottom Left"
        else:
            if y < part_height:
                return "Top Right"
            else:
                return "Bottom Right"


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()