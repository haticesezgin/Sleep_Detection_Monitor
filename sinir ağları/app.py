import tkinter as tk
import customtkinter as ctk
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
import vlc
import os

#pip install python-vlc indir
#https://www.videolan.org/vlc/ indir
os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")

app = tk.Tk()
app.geometry("600x600")
app.title("Drowsiness pop up")
ctk.set_appearance_mode("dark")

vidFrame = tk.Frame(height=480, width=600 )
vidFrame.pack()
vid = ctk.CTkLabel(vidFrame)
vid.pack()

#counter  ne kadar süre uykulu kaldık
counter = 0
counterLabel = ctk.CTkLabel(master=app, text=counter, height=40, width=120, font=("Ariel", 20), text_color="white", fg_color="teal")
counterLabel.pack(pady=10)

def reset_counter():
    global counter
    counter = 0
resetButton = ctk.CTkButton(master=app, text= "Reset Counter", command=reset_counter, height=40, width=120, font=("Ariel", 20), text_color="white", fg_color="teal")
resetButton.pack()


                       # repository adı, değiştirmeli
model = torch.hub.load('ultralytics/yolov5', 'custom', path = "model dosyasının yolu", force_reload=True)
cap = cv2.VideoCapture(3)
def detect():
    global counter
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    img = np.squeeze(results.render())

    # uyku hali tespit edilirse couter başlayacaktır
    #print(results.xywh[0]) komut koordinatları yazdırır, böylece uyanık ve uykulu olduğunuzda hangi koordinatların olduğunu bulabilirsin, böylece koordinatları değiştirebilirsin dconf & dclass
    if len(results.xywh[0]) > 0:
        dconf = results.xywh[0][0][4]
        dclass = results.xywh[0][0][5]

        if dconf.item() > 0.85 and dclass.item() == 1.0:
            #alarm dosya yolu, senin bilgisaryına göre değiştir
            p = vlc.MediaPlayer(r"file:///C:\Users\hp\FourthYearProjects\drowniness_detection\alarm-clock-1-29480.wav")
            p.play()
            counter +=1

    imgarr = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(imgarr)
    vid.imgtk = imgtk
    vid.configure(image = imgtk)
    vid.after(10, detect)
    counterLabel.configure(text = counter)


detect()
app.mainloop()