#CODE WRITTEN BY CHATGPT
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start capturing from webcam
cap = cv2.VideoCapture(0)

# Create a figure for 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the connections between landmarks
connections = mp_hands.HAND_CONNECTIONS

# Initialize Kalman filter
class KalmanFilter:
    def __init__(self):
        self.state = np.zeros((3, 1))
        self.A = np.eye(3)
        self.H = np.eye(3)
        self.Q = np.eye(3) * 0.01
        self.R = np.eye(3) * 0.1
        self.P = np.eye(3)

    def update(self, measurement):
        self.state = self.A @ self.state
        self.P = self.A @ self.P @ self.A.T + self.Q
        y = measurement.reshape(-1, 1) - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P
        return self.state.flatten()

# Create Kalman filter instance
kalman_filters = [KalmanFilter() for _ in range(21)]  # One for each landmark

# Placeholder for landmarks
landmarks = None

# Thread to capture video
def capture_video():
    global landmarks
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]  # Track the first detected hand
        else:
            landmarks = None

# Start video capture in a separate thread
video_thread = threading.Thread(target=capture_video)
video_thread.start()

# Function to update the 3D plot
def update_plot():
    ax.cla()  # Clear previous plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Hand Landmarks')

    if landmarks:
        filtered_landmarks = []
        for i, landmark in enumerate(landmarks.landmark):
            measurement = np.array([landmark.x, landmark.y, landmark.z])
            filtered_position = kalman_filters[i].update(measurement)
            filtered_landmarks.append(filtered_position)

        filtered_landmarks = np.array(filtered_landmarks)
        xs, ys, zs = filtered_landmarks[:, 0], filtered_landmarks[:, 1], filtered_landmarks[:, 2]
        
        ax.scatter(xs, ys, zs, s=100)

        # Draw lines between the joints
        for connection in connections:
            x = [xs[connection[0]], xs[connection[1]]]
            y = [ys[connection[0]], ys[connection[1]]]
            z = [zs[connection[0]], zs[connection[1]]]
            ax.plot(x, y, z, color='blue')

        # Center the plot on the hand
        ax.set_xlim([xs.min(), xs.max()])
        ax.set_ylim([ys.min(), ys.max()])
        ax.set_zlim([zs.min(), zs.max()])

# Animation function to update the 3D plot continuously
def animate(i):
    update_plot()

# Create an animation with reduced frame rate
ani = FuncAnimation(fig, animate, interval=100)  # You can increase interval for lower frame rate

plt.show()

cap.release()
video_thread.join()