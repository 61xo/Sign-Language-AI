import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model


model = load_model('model_dl.keras')
scaler = joblib.load('scaler.pkl')
classes = np.load('deep_classes.npy', allow_pickle=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    data_aux = []
    
    ret, frame = cap.read()
    if not ret:
        break
    
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


            for i in range(len(hand_landmarks.landmark)):
                
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z 
                
                data_aux.append(x)
                data_aux.append(y)
                data_aux.append(z)
            # ---------------------------------------------------------

            if len(data_aux) == 63: 
                input_data = np.array(data_aux).reshape(1, -1)
                
              
                input_scaled = scaler.transform(input_data)
                
                prediction = model.predict(input_scaled, verbose=0)
                
                max_prob = np.max(prediction)
                class_index = np.argmax(prediction)
                predicted_char = classes[class_index]

                
                if max_prob > 0.80:
                    text = f"{predicted_char} {max_prob*100:.1f}%"
                    color = (0, 255, 0)
                else:
                    text = "..."
                    color = (0, 0, 255)

              
                x1 = int(hand_landmarks.landmark[0].x * W)
                y1 = int(hand_landmarks.landmark[0].y * H) - 20
                cv2.putText(frame, text, (x1, y1), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

    cv2.imshow('AI Sign Language', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()