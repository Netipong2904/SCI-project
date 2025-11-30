import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'space', 27: 'DEL', 28: 'Cls'
}

window = tk.Tk()
window.title("Hand Gesture Recognition")

label = tk.Label(window)
label.pack()

# Create a text box for predictions
text_box = tk.Text(window, height=2, width=60, font=('Arial', 24))
text_box.pack()

last_prediction = None
last_update_time = datetime.now()
predictions = []
no_input_time = datetime.now()
del_time = None

def update_frame():
    global last_prediction, last_update_time, predictions, no_input_time, del_time

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        return

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        no_input_time = datetime.now() 

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            x_min = min(x_)
            y_min = min(y_)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - x_min)
                data_aux.append(y - y_min)

        data_aux = data_aux[:84] + [0] * (84 - len(data_aux))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        if results.multi_hand_landmarks:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame_rgb, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        current_time = datetime.now()
        if predicted_character == 'DEL':
            if del_time is None:
                del_time = current_time
            elif (current_time - del_time).seconds >= 3:  
                if predictions:
                    predictions.pop() 
                del_time = None  
        else:
            del_time = None 


        if predicted_character != last_prediction:
            last_prediction = predicted_character
            last_update_time = current_time
        elif (current_time - last_update_time).seconds >= 2:
            if last_prediction is not None and last_prediction != 'DEL':
                if predicted_character == 'space':
                    predictions.append(' ') 
                else:
                    predictions.append(last_prediction)
                last_update_time = current_time

        predictions_text = ''.join(predictions)
        text_box.delete('1.0', tk.END)
        text_box.insert(tk.END, predictions_text)

    else:
        no_input_time = datetime.now() 

    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    window.after(10, update_frame)

update_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()

