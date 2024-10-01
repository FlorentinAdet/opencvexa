#Faire 'pip install opencv-python mediapipe numpy' si jamais ça ne marche pas

import cv2
import mediapipe as mp
import numpy as np

# Initialisation de Mediapipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Ouverture Webcam 
video = cv2.VideoCapture(0)

def recognize_gesture(hand_landmarks):
    # Récupérer les positions des points de repère de la main
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Définir des seuils simples pour la reconnaissance
    thumb_open = thumb_tip.y < index_tip.y
    fingers_open = index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y and ring_tip.y < pinky_tip.y

    if thumb_open and fingers_open:
        return "Poing fermé"
    else:
        return "Main ouverte"

while True:
    # Lire l'image suivante
    ret, frame = video.read()
    
    if not ret:
        break

    # Convertir l'image en RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Traitement de l'image pour la détection des mains
    results = hands.process(rgb_frame)

    # Vérifier si des mains sont détectées
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les repères de la main
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Reconnaître le geste
            gesture = recognize_gesture(hand_landmarks)
            cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher le résultat
    cv2.imshow("Détection des mains et reconnaissance de gestes", frame)

    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video.release()
cv2.destroyAllWindows()
