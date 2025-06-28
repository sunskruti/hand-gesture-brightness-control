# Hand Gesture-Based Brightness Control 👋💡

A real-time computer vision project that allows users to control screen brightness using hand gestures, powered by OpenCV and MediaPipe.

## 📅 Duration
**March 2023 – April 2023**

## 🚀 Overview
This personal project implements a gesture-controlled brightness adjustment system using a webcam and Python libraries. No external hardware or keyboard input is needed!

## 🎯 Key Features
- Track hand landmarks via webcam input.
- Map finger distance to screen brightness level.
- Smooth and accurate real-time gesture recognition using OpenCV and Google MediaPipe.
- Adjust brightness dynamically through gestures using PyAutoGUI.

## 🛠️ Technologies Used
- Python
- OpenCV
- MediaPipe
- PyAutoGUI

## 📷 Demo
> *(Include a short GIF or video demonstrating the functionality. Upload it to the repo or link externally.)*

## 🧠 How It Works
1. Capture real-time webcam feed.
2. Use MediaPipe to detect and extract hand landmarks.
3. Calculate distance between thumb and index finger.
4. Convert the distance to a brightness level.
5. Use PyAutoGUI to adjust system brightness accordingly.

## 🔄 Future Improvements
- Multi-hand detection and control.
- Cross-platform brightness adjustment (currently OS-dependent).
- GUI interface for toggling gesture mode.

## 📁 Project Structure

