#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Anmol
#
# Created:     15-04-2024
# Copyright:   (c) Anmol 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import serial
import torch
from torchvision import transforms
from ultralytics import YOLO

# Load YOLOv5 model
model = torch.load('best.pt', map_location=torch.device('cpu'))  # Load the model file
model.eval()    

# Function to read sensor data from Arduino
def read_sensor_data(ser):
    # Assuming Arduino sends sensor data as a single line of comma-separated values
    data = ser.readline().decode().strip().split(',')
    return data

# Function to update sensor data on the GUI
def update_sensor_data():
    # Read sensor data from Arduino
    sensor_data = read_sensor_data(ser)

    # Update sensor data labels
    sensor_label.config(text="Sensor Data: {}".format(sensor_data))

# Function to update camera feed with object detection
def update_camera_feed():
    # Capture frame from webcam
    ret, frame = cap.read()

    # Perform object detection on the frame using YOLOv5
    results = model(frame)

    # Process detection results
    # Draw bounding boxes on the frame
    for result in results:
        x1, y1, x2, y2, class_id, confidence = result
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_id}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert frame to format suitable for Tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)

    # Update camera feed label
    camera_label.img = img
    camera_label.config(image=img)

    # Schedule next update
    camera_label.after(10, update_camera_feed)

# Main function to create GUI
def main():
    # Create main window
    root = tk.Tk()
    root.title("Object Detection and Sensor Data")

    # Create camera feed label
    camera_label = tk.Label(root)
    camera_label.pack()

    # Create sensor data label
    sensor_label = tk.Label(root, text="Sensor Data: ")
    sensor_label.pack()

    # Create serial connection to Arduino
    ser = serial.Serial('COM3', 9600)  # Change COM port as necessary

    # Start webcam capture
    cap = cv2.VideoCapture(0)  # Use webcam at index 0

    # Start updating camera feed and sensor data
    update_camera_feed()
    update_sensor_data()

    # Run GUI main loop
    root.mainloop()

if __name__ == "__main__":
    main()
