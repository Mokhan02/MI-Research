import cv2
from ultralytics import YOLO
from PIL import Image
import tkinter as tk

model = YOLO("yolov8n.pt")  # downloads automatically, n = nano (fast)

def show_image():
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    img = Image.open("aayush.jpg")  # your meme image
    img = img.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
    from PIL import ImageTk
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=photo)
    label.pack()
    root.after(3000, root.destroy)  # close after 3 seconds
    root.mainloop()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame, verbose=False)
    
    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls)] == "cell phone" and float(box.conf) > 0.5:
                print("Dont test me bro")
                show_image()  # trigger meme
    
    cv2.imshow("Studying...", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()