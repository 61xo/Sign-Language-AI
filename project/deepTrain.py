import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1) Load Data
data = pd.read_csv("data1.csv")

labels = data.iloc[:, 0]
features = data.iloc[:, 1:].values 

# 2) Preprocessing

# a) Encoding Labels:
label_encoder = LabelEncoder()
y_integer = label_encoder.fit_transform(labels)
#  One-Hot Encoding
y_categorical = to_categorical(y_integer)

# b) Scaling Features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(features)

# c) Splitting: 
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_categorical, test_size=0.2, random_state=42)

# 3) Build Deep Learning Model (The Architecture)
model = Sequential()

# Input Layer + 1st Hidden Layer
# units=128
model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3)) 

# 2nd Hidden Layer
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))

# 3rd Hidden Layer
model.add(Dense(units=32, activation='relu'))

# Output Layer
# Softmax
model.add(Dense(units=y_categorical.shape[1], activation='softmax'))

# 4) Compile Model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 5) Train Model
# EarlyStopping:
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

print("Starting Training...")
history = model.fit(X_train, y_train, 
                    epochs=200,            
                    batch_size=32,          
                    validation_data=(X_test, y_test), 
                    callbacks=callbacks)

# 6) Evaluate & Save
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

model.save("model_dl.keras")
np.save("deep_classes.npy", label_encoder.classes_)
joblib.dump(scaler, "scaler.pkl") 

print("Deep Learning Model Saved Successfully!")