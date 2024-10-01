import cv2

# Choisissez l'algorithme de suivi
tracker = cv2.TrackerCSRT_create()  # Vous pouvez aussi utiliser cv2.TrackerKCF_create()

# Ouvrir la vidéo (0 pour la webcam)
video = cv2.VideoCapture(0)

# Lire la première image
ret, frame = video.read()

# Sélectionner la région d'intérêt (ROI)
roi = cv2.selectROI("Sélectionner l'objet", frame, fromCenter=False, showCrosshair=True)

# Initialiser le suivi
tracker.init(frame, roi)

while True:
    # Lire l'image suivante
    ret, frame = video.read()
    
    if not ret:
        break

    # Mettre à jour le tracker
    success, box = tracker.update(frame)

    # Dessiner le rectangle autour de l'objet suivi
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Objet perdu", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Afficher le résultat
    cv2.imshow("Suivi d'objet", frame)

    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video.release()
cv2.destroyAllWindows()
