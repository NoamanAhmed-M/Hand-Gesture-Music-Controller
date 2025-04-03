
import cv2
import mediapipe as mp
import math
import numpy as np
import pygame
from pydub import AudioSegment
from pydub.playback import play
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Get the absolute path to the song file
script_dir = os.path.dirname(os.path.abspath(__file__))
song_path = os.path.join(script_dir, "alone.mp3")

# Verify the file exists
if not os.path.exists(song_path):
    raise FileNotFoundError(f"Audio file not found at: {song_path}")

# Load the song
original_song = AudioSegment.from_file(song_path, format="mp3")

# Initialize pygame for audio playback
pygame.mixer.init()
try:
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play(-1)  # Loop indefinitely
    pygame.mixer.music.set_volume(0.5)  # Start with medium volume
except pygame.error as e:
    print(f"Error loading audio file: {e}")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening webcam")
    exit()

def calculate_distance(p1, p2, w, h):
    """Calculate Euclidean distance between two points"""
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        h, w, _ = frame.shape

        # Store hand landmarks if detected
        hands_landmarks = []
        if results.multi_hand_landmarks:
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hands_landmarks.append(hand_landmarks)

        # Adjust audio parameters if both hands are detected
        if len(hands_landmarks) >= 2:
            hand1 = hands_landmarks[0]
            hand2 = hands_landmarks[1]
            
            # Speed: Distance between thumb tip (4) and middle finger tip (8) of first hand
            speed_distance = calculate_distance(hand1.landmark[4], hand1.landmark[8], w, h)
            speed_factor = np.interp(speed_distance, [20, 200], [0.5, 2.0])  # Speed between 0.5x and 2.0x
            
            # Display speed factor on screen
            cv2.putText(frame, f"Speed: {speed_factor:.2f}x", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Volume: Distance between middle finger tips (8) of both hands
            volume_distance = calculate_distance(hand1.landmark[8], hand2.landmark[8], w, h)
            volume_level = np.interp(volume_distance, [50, 400], [0.1, 1.0])  # Volume 10% to 100%
            pygame.mixer.music.set_volume(volume_level)
            
            # Display volume level on screen
            cv2.putText(frame, f"Volume: {int(volume_level * 100)}%", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Pitch: Distance between thumb tip (4) and middle finger tip (8) of second hand
            pitch_distance = calculate_distance(hand2.landmark[4], hand2.landmark[8], w, h)
            pitch_factor = np.interp(pitch_distance, [20, 200], [0.8, 1.2])  # Pitch from 0.8x to 1.2x
            
            # Display pitch factor on screen
            cv2.putText(frame, f"Pitch: {pitch_factor:.2f}x", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            try:
                # Adjust pitch
                song = original_song._spawn(original_song.raw_data, overrides={
                    "frame_rate": int(original_song.frame_rate * pitch_factor)
                })
                temp_song_path = os.path.join(script_dir, "temp_song.mp3")
                song.export(temp_song_path, format="mp3")
                
                # Only reload if the pitch has changed significantly
                if abs(pitch_factor - getattr(pygame.mixer.music, "_last_pitch_factor", 0)) > 0.05:
                    pygame.mixer.music._last_pitch_factor = pitch_factor
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(temp_song_path)
                    pygame.mixer.music.play(-1)
            except Exception as e:
                print(f"Error adjusting pitch: {e}")
        else:
            # Display instructions if fewer than 2 hands are detected
            cv2.putText(frame, "Show both hands to control music", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Hand Music Controller", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    # Clean up temporary file if it exists
    temp_song_path = os.path.join(script_dir, "temp_song.mp3")
    if os.path.exists(temp_song_path):
        os.remove(temp_song_path)