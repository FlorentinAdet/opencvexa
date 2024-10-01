import cv2
import numpy as np
import os

# Modèles pré-entraînés pour la détection de visages, l'estimation d'âge et la classification de genre
face_model = os.getcwd()+'/Detection Visage avec CNN/models/deploy.caffemodel'
face_proto = os.getcwd()+'/Detection Visage avec CNN/models/deploy.prototxt'
age_model = os.getcwd()+'/Detection Visage avec CNN/models/age_net.caffemodel'
age_proto = os.getcwd()+'/Detection Visage avec CNN/models/age_deploy.prototxt'
gender_model = os.getcwd()+'/Detection Visage avec CNN/models/gender_net.caffemodel'
gender_proto = os.getcwd()+'/Detection Visage avec CNN/models/gender_deploy.prototxt'
    


print("Répertoire de travail actuel :", os.getcwd())

# Charger les réseaux DNN
face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# Plages d'âge et labels de genre
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Dimensions pour la détection des visages
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Fonction pour la détection des visages et évaluation du genre et de l'âge
def detect_and_predict(frame):
    h, w = frame.shape[:2]
    
    # Préparer l'image pour la détection de visages
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Seuil de confiance
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # Extraire le visage détecté
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            # Préparer le visage pour la prédiction du genre
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Prédiction du genre
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            
            # Prédiction de l'âge
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            # Dessiner un rectangle autour du visage et afficher l'âge et le genre
            label = f'{gender}, {age}'
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame

# Flux vidéo à partir de la webcam
cap = cv2.VideoCapture(0)

# Vérification que le dossier d'enregistrement existe, sinon le créer
output_dir = 'images/exo3'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Variable pour contrôler si une capture a déjà été faite
screenshot_taken = False

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Erreur lors de la capture vidéo.")
        break

    # Appliquer la détection de visage, genre et âge
    frame = detect_and_predict(frame)

    # Afficher le flux vidéo en temps réel
    cv2.imshow("Détection de visages, genre et âge", frame)

    # Sauvegarder une capture d'écran d'une détection réussie
    if not screenshot_taken:
        cv2.imwrite(os.path.join(output_dir, 'capture_detection.png'), frame)
        screenshot_taken = True

    # Quitter la boucle en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
