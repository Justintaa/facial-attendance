import tkinter as tk
from tkinter import simpledialog
import cv2
import threading
import face_recognition
import os
import pandas as pd
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
import pickle  

DATA_FILE = "face_data.pkl"
known_faces_dir = 'data/known'
known_face_encodings = []
known_face_names = []

# Function to load existing face data from the pickle file if it exists
def load_face_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as file:
            return pickle.load(file)
    return [], []

# Load known face encodings from the directory if they haven't been loaded
known_face_encodings, known_face_names = load_face_data()

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.png', 'jpeg')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(filename.split('.')[0])

class AttendanceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Facial Detection Attendance System")

        self.start_button = tk.Button(master, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="Stop Recognition", command=self.stop_recognition)
        self.stop_button.pack(pady=10)

        self.canvas = tk.Label(master)
        self.canvas.pack()

        self.running = False
        self.video_thread = None
        self.recognized_faces = set()
        self.prompted_faces = {}
        self.recent_logs = {}

        self.my_name = "Justin"  

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_recognition(self):
        if not self.running:
            self.running = True
            self.video_thread = threading.Thread(target=self.capture_video)
            self.video_thread.start()

    def stop_recognition(self):
        self.running = False

    def on_closing(self):
        self.stop_recognition()
        self.master.destroy()

    def capture_video(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the camera.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                if tuple(face_encoding) in self.recognized_faces:
                    continue

                if self.is_face_being_prompted(face_encoding):
                    continue

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    self.recognized_faces.add(tuple(face_encoding))

                    if name == self.my_name:
                        self.log_attendance(name)

                else:
                    self.master.after(0, self.get_name_from_user, face_encoding)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.update_canvas(frame)

        cap.release()

    def is_face_being_prompted(self, face_encoding):
        current_time = datetime.now().timestamp()
        self.prompted_faces = {tuple(enc): ts for enc, ts in self.prompted_faces.items() if current_time - ts < 5}
        for known_enc in self.prompted_faces:
            if face_recognition.compare_faces([np.array(known_enc)], face_encoding, tolerance=0.6)[0]:
                return True
        return False

    def get_name_from_user(self, face_encoding):
        self.prompted_faces[tuple(face_encoding)] = datetime.now().timestamp()
        name = simpledialog.askstring("Input", "Enter your name:")
        if name:
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            self.recognized_faces.add(tuple(face_encoding))
            self.log_attendance(name)
            self.save_face_data(known_face_encodings, known_face_names)

    def update_canvas(self, frame):
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        self.canvas.config(image=photo)
        self.canvas.image = photo

    def log_attendance(self, name):
        current_time = datetime.now().timestamp()
        if name in self.recent_logs and current_time - self.recent_logs[name] < 300:
            return  

        self.recent_logs[name] = current_time

        log_file = 'logs/attendance.csv'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
            df = pd.DataFrame(columns=['Name', 'Timestamp'])
            df.to_csv(log_file, index=False)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame({'Name': [name], 'Timestamp': [timestamp]})], ignore_index=True)
        df.to_csv(log_file, index=False)

    def save_face_data(self, encodings, names):
        print("Saving updated face data")
        with open(DATA_FILE, 'wb') as file:
            pickle.dump((encodings, names), file)

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()

