import cv2
import os
import numpy as np
import pandas as pd
from deepface import DeepFace
from mtcnn import MTCNN
from time import sleep

DATASET_DIR = 'user_faces'
EMBEDDINGS_FILE = 'all_embeddings.csv'
FACE_SIZE = (112, 112)
REQUIRED_SAMPLES = 100
CONFIDENCE_THRESHOLD = 0.65

os.makedirs(DATASET_DIR, exist_ok=True)

def open_webcam():
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Opened webcam at index {idx}")
            return cap
        cap.release()
    raise RuntimeError("No webcam detected. Check connection and OS permissions.")

def load_all_embeddings_labels():
    all_embeddings, all_labels = [], []
    for user in os.listdir(DATASET_DIR):
        emb_file = os.path.join(DATASET_DIR, user, f"{user}_embeddings.csv")
        if os.path.exists(emb_file):
            df = pd.read_csv(emb_file)
            labels = df['label'].tolist()
            embeddings = df.drop(columns=['label']).values.astype(np.float32)
            all_embeddings.extend(embeddings)
            all_labels.extend(labels)
    if all_embeddings:
        return np.array(all_embeddings), all_labels
    else:
        return np.array([]), []

def is_duplicate_face(new_embeddings, all_embeddings, all_labels, candidate_name, sim_threshold=CONFIDENCE_THRESHOLD):
    from numpy.linalg import norm
    for emb in new_embeddings:
        for known_emb, label in zip(all_embeddings, all_labels):
            csim = np.dot(emb, known_emb) / (norm(emb) * norm(known_emb) + 1e-8)
            if csim >= sim_threshold and label != candidate_name:
                return label
    return None

def register_user(name):
    print(f"\nStarting registration for user: {name}")
    user_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(user_dir, exist_ok=True)
    detector = MTCNN()
    cap = open_webcam()
    count = 0
    samples = []
    print(f"Collecting {REQUIRED_SAMPLES} face samples. Please look at the camera from different angles and expressions.")
    while count < REQUIRED_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector.detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            face_img = cv2.resize(face_img, FACE_SIZE)
            file_path = os.path.join(user_dir, f"{name}_{count}.jpg")
            cv2.imwrite(file_path, face_img)
            samples.append(face_img)
            count += 1
        display = f'Samples: {count}/{REQUIRED_SAMPLES}'
        cv2.putText(frame, display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Registration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            # Cleanup partial files and folder
            for f in os.listdir(user_dir):
                os.remove(os.path.join(user_dir, f))
            os.rmdir(user_dir)
            return False
    cap.release()
    cv2.destroyAllWindows()

    print(f"Captured {count} samples for user {name}.")

    # Compute embeddings for samples
    embeddings = []
    for idx in range(count):
        img_path = os.path.join(user_dir, f"{name}_{idx}.jpg")
        try:
            rep = DeepFace.represent(img_path=img_path, model_name='ArcFace')[0]['embedding']
            embeddings.append(np.array(rep))
        except Exception as e:
            print(f"Embedding error for {img_path}: {e}")

    new_embeddings = np.array(embeddings)
    all_embeddings, all_labels = load_all_embeddings_labels()
    if all_embeddings.size > 0:
        dup_label = is_duplicate_face(new_embeddings, all_embeddings, all_labels, name)
        if dup_label:
            print(f"Face matches an existing user: '{dup_label}'. Registration aborted.")
            # Cleanup user files if duplicate
            for f in os.listdir(user_dir):
                os.remove(os.path.join(user_dir, f))
            os.rmdir(user_dir)
            return False

    # Save embeddings
    user_df = pd.DataFrame(embeddings)
    user_df['label'] = name
    emb_file = os.path.join(user_dir, f"{name}_embeddings.csv")
    user_df.to_csv(emb_file, index=False)
    print(f"User embeddings saved to {emb_file}")
    print(f"User '{name}' registered successfully.")
    return True

def aggregate_all_embeddings():
    all_embeddings = []
    for user in os.listdir(DATASET_DIR):
        emb_file = os.path.join(DATASET_DIR, user, f"{user}_embeddings.csv")
        if os.path.exists(emb_file):
            df = pd.read_csv(emb_file)
            all_embeddings.append(df)
    if all_embeddings:
        all_emb_df = pd.concat(all_embeddings, ignore_index=True)
        all_emb_df.to_csv(EMBEDDINGS_FILE, index=False)
        print(f"Aggregated embeddings saved as {EMBEDDINGS_FILE}")
        return all_emb_df
    else:
        print("No embeddings found for any user.")
        return pd.DataFrame()

def recognize_realtime():
    emb_df = aggregate_all_embeddings()
    if emb_df.empty:
        print("No embeddings to recognize against. Register user(s) first.")
        return
    known_embeddings = emb_df.drop('label', axis=1).values.astype(np.float32)
    labels = emb_df['label'].values.tolist()
    detector = MTCNN()
    cap = open_webcam()
    from numpy.linalg import norm
    print("Recognition started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera frame.")
            break
        faces = detector.detect_faces(frame)
        displayed = "No face detected"
        if faces:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                displayed = "Crop error"
            else:
                try:
                    face_img = cv2.resize(face_img, FACE_SIZE)
                    temp_img = "__temp.jpg"
                    cv2.imwrite(temp_img, face_img)
                    emb = DeepFace.represent(img_path=temp_img, model_name='ArcFace')[0]["embedding"]
                    emb = np.array(emb).astype(np.float32)
                    scores = np.dot(known_embeddings, emb) / (norm(known_embeddings, axis=1) * norm(emb) + 1e-8)
                    idx = np.argmax(scores)
                    if scores[idx] > CONFIDENCE_THRESHOLD:
                        displayed = f"{labels[idx]} ({scores[idx]:.2f})"
                    else:
                        displayed = "Unknown"
                    os.remove(temp_img)
                except Exception as e:
                    displayed = f"Recognition error"
        cv2.putText(frame, displayed, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,80,0), 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("== Face Registration and Recognition ==")
    while True:
        try:
            opt = input("\n1. Register new user\n2. Recognize faces\n3. Exit\nSelect option (1/2/3): ").strip()
            if opt == "1":
                name = input("Enter new user name: ").strip()
                if not name:
                    print("Blank name. Try again.")
                    continue
                success = register_user(name)
                if not success:
                    print("Registration failed or cancelled. Try again.")
                    continue
            elif opt == "2":
                recognize_realtime()
            elif opt == "3":
                print("Finished.")
                break
            else:
                print("Invalid option. Choose 1, 2, or 3.")
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
