import mediapipe as mp
import cv2
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

dataset_path = "//home//omarxo//coding//py/AI//dataset"
output_file = "data1.csv"

data = []

for label in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, label)

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            
            row = [label]
            for p in lm.landmark:
                row += [p.x, p.y, p.z]

            data.append(row)


with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("Done collecting landmarks!")
