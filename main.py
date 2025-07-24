import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Settings
csv_filename = 'face_dataset.csv'
face_size = (100, 100)
face_size_flat = face_size[0]*face_size[1]
capture_time = 20  # seconds

# Helper: Load previous dataset if it exists
if os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename)
    existing_labels = set(df['label'])
    existing_X = df[[f'pixel_{i}' for i in range(face_size_flat)]].values
    existing_y = df['label'].values
else:
    df = None
    existing_labels = set()
    existing_X = np.array([])
    existing_y = np.array([])

# Ask for new name and check for duplicate
while True:
    name = input("Enter your name: ").strip()
    if name in existing_labels:
        print("This name is already registered! Try another name.")
    elif len(name) == 0:
        print("Name cannot be blank.")
    else:
        break

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

print("Look into the camera. Face recording will run for 20 seconds...")
cap = cv2.VideoCapture(0)
faces = []

start = time.time()
while time.time() - start < capture_time:
    ret, frame = cap.read()
    if not ret:
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            xmin = max(int(bbox.xmin * w), 0)
            ymin = max(int(bbox.ymin * h), 0)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            face_img = frame[ymin:ymin+height, xmin:xmin+width]
            if face_img.size != 0:
                face_img_resized = cv2.resize(face_img, face_size)
                gray_face = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2GRAY)
                faces.append(gray_face.flatten())
    cv2.putText(frame, f"Recording... {int(capture_time - (time.time() - start))}s left", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit("User exited during recording.")

cap.release()
cv2.destroyAllWindows()

if len(faces) == 0:
    exit("No faces captured. Try again with better lighting/positioning!")

faces = np.array(faces)
labels = np.array([name]*len(faces))

# 1. If there ARE existing users, check for face duplicates before adding this user.
if existing_X.size:
    # Train a simple model with stored faces
    knn_check = KNeighborsClassifier(n_neighbors=1)
    knn_check.fit(existing_X, existing_y)
    # Predict current face batches
    preds = knn_check.predict(faces)
    # If any of the captured faces match an existing (non-random) user
    for pred_label in preds:
        if pred_label != "not_"+name and pred_label in existing_labels:
            print(f"Your face matches an already registered user: '{pred_label}'. Registration aborted.")
            exit()
# Otherwise: process as new user

# 2. Prepare synthetic negatives for training
neg_faces = np.random.randint(0, 255, (len(faces), face_size_flat), dtype=np.uint8)
neg_labels = np.array(['not_'+name]*len(faces))
X = np.concatenate([faces, neg_faces])
y = np.concatenate([labels, neg_labels])

# 3. Save/append to CSV
header = ['label'] + [f'pixel_{i}' for i in range(face_size_flat)]
save_mode = 'a' if os.path.exists(csv_filename) else 'w'
with open(csv_filename, mode=save_mode, newline='') as file:
    writer = csv.writer(file)
    if save_mode == 'w':
        writer.writerow(header)
    for label, row_pixels in zip(y, X):
        writer.writerow([label] + list(row_pixels))

print(f"Dataset updated and saved to {csv_filename}")

# 4. Combine *all* (including new) data for model training
if existing_X.size:
    X_full = np.concatenate([existing_X, X])
    y_full = np.concatenate([existing_y, y])
else:
    X_full, y_full = X, y

# 5. Train model
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"AI model trained. Recognition accuracy: {acc*100:.2f}%")

# 6. Real-time recognition with all users
cap = cv2.VideoCapture(0)
print("Show your face. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    recognized = "No face detected"
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            xmin = max(int(bbox.xmin * w), 0)
            ymin = max(int(bbox.ymin * h), 0)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            face_img = frame[ymin:ymin+height, xmin:xmin+width]
            if face_img.size != 0:
                face_img_resized = cv2.resize(face_img, face_size)
                gray_face = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2GRAY).flatten().reshape(1, -1)
                pred = knn.predict(gray_face)[0]
                proba = knn.predict_proba(gray_face)[0]
                idx = knn.classes_.tolist().index(pred)
                confidence = proba[idx]
                recognized = f"{pred} (confidence: {confidence*100:.2f}%)"
    cv2.putText(frame, recognized, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

