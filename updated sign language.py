from tkinter import *
from tkinter import ttk
from googletrans import Translator, LANGUAGES
import speech_recognition as sr
import threading
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands model for sign language recognition
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Translator instances for multiple APIs
google_translator = Translator()

# Function to recognize basic sign language using MediaPipe (ASL Alphabet)
def sign_language_recognition():
    cap = cv2.VideoCapture(0)  # Open webcam
    recognized_text = ""
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB and process it with MediaPipe Hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Call the function to detect the sign language gesture
                recognized_text = detect_sign_language(hand_landmarks)
                
                # Update the input text with recognized gesture
                Input_text.delete(1.0, END)
                Input_text.insert(END, recognized_text)
                Translate()  # Automatically translate the recognized sign language text
        
        # Show webcam feed with hand landmarks
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to map hand landmarks to basic ASL gestures (e.g., 'A', 'B', 'C', etc.)
def detect_sign_language(hand_landmarks):
    # Extract relevant landmarks for the fingers
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # Get distances between specific landmarks to define gestures
    index_thumb_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    middle_wrist_distance = np.linalg.norm(np.array([middle_tip.x, middle_tip.y]) - np.array([wrist.x, wrist.y]))
    pinky_wrist_distance = np.linalg.norm(np.array([pinky_tip.x, pinky_tip.y]) - np.array([wrist.x, wrist.y]))

    # Define gestures based on relative positions of the fingers
    if index_thumb_distance < 0.05 and pinky_wrist_distance > 0.2:
        return "A"  # Example: Thumb and index finger are close, pinky is far (basic 'A' shape)
    elif index_thumb_distance > 0.1 and middle_wrist_distance < 0.2:
        return "B"  # Example: All fingers extended upward
    elif middle_wrist_distance < 0.1 and index_thumb_distance > 0.1:
        return "C"  # Example: C-shape made by hand
    else:
        return "Unknown Gesture"  # Fallback if gesture is not recognized

# Function to convert speech to text
def real_time_speech_translation():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)  # Speech to text
                Input_text.delete(1.0, END)
                Input_text.insert(END, text)
                
                # Automatically trigger translation after recognizing speech
                Translate()
                
            except Exception as e:
                print(f"Error: {e}")
                Input_text.insert(END, "Sorry, I could not recognize your voice.")
                break

# Translator function with multiple API options
def Translate():
    input_text = Input_text.get(1.0, END).strip()
    
    if not input_text:
        Output_text.delete(1.0, END)
        Output_text.insert(END, "Please enter some text or speak.")
        return

    # Fetch user selection for translation API
    api_choice = api_selector.get()

    if api_choice == "Google Translate":
        translated = google_translator.translate(text=input_text, src=src_lang.get(), dest=dest_lang.get())
        translated_text = translated.text
    
    elif api_choice == "Other Translator":  # Placeholder for another API
        # translated = other_translator.translate(text=input_text, src_lang=src_lang.get(), dest_lang=dest_lang.get())
        # translated_text = other_translator['translation']
        translated_text = "Translation from other API (hypothetical)"
    
    else:
        translated_text = "Please select a valid API."
    
    Output_text.delete(1.0, END)
    Output_text.insert(END, translated_text)

# GUI setup
root = Tk()
root.geometry('1080x500')
root.resizable(0, 0)
root.title("Real-Time Language Translator with Sign Language")
root.config(bg='ghost white')

# Heading
Label(root, text="REAL-TIME LANGUAGE TRANSLATOR", font="arial 20 bold", bg='white smoke').pack()
Label(root, text="Language Translator", font='arial 20 bold', bg='white smoke', width='20').pack(side='bottom')

# Input and Output Text Widgets
Label(root, text="Enter or Speak Text / Sign Language", font='arial 13 bold', bg='white smoke').place(x=200, y=60)
Input_text = Text(root, font='arial 10', height=11, wrap=WORD, padx=5, pady=5, width=60)
Input_text.place(x=30, y=100)

Label(root, text="Translated Output", font='arial 13 bold', bg='white smoke').place(x=780, y=60)
Output_text = Text(root, font='arial 10', height=11, wrap=WORD, padx=5, pady=5, width=60)
Output_text.place(x=600, y=100)

# Language Selection
language = list(LANGUAGES.values())
src_lang = ttk.Combobox(root, values=language, width=22)
src_lang.place(x=20, y=60)
src_lang.set('choose input language')

dest_lang = ttk.Combobox(root, values=language, width=22)
dest_lang.place(x=890, y=60)
dest_lang.set('choose output language')

# API Selection
api_selector = ttk.Combobox(root, values=["Google Translate", "Other Translator"], width=22)
api_selector.place(x=490, y=250)
api_selector.set('Choose API')

# Buttons
trans_btn = Button(root, text='Translate', font='arial 12 bold', pady=5, command=Translate, bg='royal blue1', activebackground='sky blue')
trans_btn.place(x=490, y=180)

# Real-Time Speech Translation Button
real_time_speech_btn = Button(root, text='Start Real-Time Speech Translation', font='arial 12 bold', pady=5, command=lambda: threading.Thread(target=real_time_speech_translation).start(), bg='light green', activebackground='green')
real_time_speech_btn.place(x=450, y=300)

# Sign Language Translation Button
sign_language_btn = Button(root, text='Start Sign Language Translation', font='arial 12 bold', pady=5, command=lambda: threading.Thread(target=sign_language_recognition).start(), bg='light blue', activebackground='blue')
sign_language_btn.place(x=450, y=350)

root.mainloop()
