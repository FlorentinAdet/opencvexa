import cv2
import numpy as np
import os

# Taille du carré ROI
ROI_SIZE = 400

# Variable pour stocker les coordonnées du clic
clic_coords = None

# Fonction de rappel pour gérer le clic de souris
def on_mouse(event, x, y, flags, param):
    global clic_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        # Lorsque l'utilisateur clique, enregistrer les coordonnées du centre du ROI
        clic_coords = (x, y)

# Fonction pour augmenter la luminosité d'une image (ROI)
def augmenter_luminosite(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convertir en espace HSV
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)  # Augmenter la valeur de la composante V (luminosité)
    v = np.clip(v, 0, 255)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Charger l'image
image_path = os.getcwd()+'/Masquage et ROI/images/image.jpg'
img = cv2.imread(image_path)

if img is None:
    print("L'image n'a pas pu être chargée. Vérifiez le chemin spécifié.")
    exit()

# Afficher l'image et configurer le gestionnaire de clic
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", on_mouse)

# Attendre que l'utilisateur clique pour définir le ROI
while True:
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF

    # Si un clic a été effectué
    if clic_coords:
        x, y = clic_coords

        # Calculer les coordonnées du ROI (centré sur le point cliqué)
        x1 = max(0, x - ROI_SIZE // 2)
        y1 = max(0, y - ROI_SIZE // 2)
        x2 = min(img.shape[1], x1 + ROI_SIZE)
        y2 = min(img.shape[0], y1 + ROI_SIZE)

        # Extraire la ROI
        roi_img = img[y1:y2, x1:x2]

        # Appliquer une modification à la ROI (par exemple, augmenter la luminosité)
        roi_modifiee = augmenter_luminosite(roi_img, value=50)

        # Remplacer la ROI dans l'image d'origine par la ROI modifiée
        img[y1:y2, x1:x2] = roi_modifiee

        # Créer un masque pour flouter uniquement l'extérieur de la ROI
        masque = np.zeros(img.shape[:2], dtype="uint8")
        cv2.rectangle(masque, (x1, y1), (x2, y2), 255, -1)

        # Appliquer le flou gaussien à toute l'image
        img_flou = cv2.GaussianBlur(img, (51, 51), 0)

        # Combiner l'image floue et la ROI non floutée
        img_resultat = cv2.bitwise_and(img_flou, img_flou, mask=cv2.bitwise_not(masque))
        img_resultat[y1:y2, x1:x2] = roi_modifiee

        # Afficher le résultat
        cv2.imshow("Image modifiée", img_resultat)

        # Sauvegarder l'image finale
        cv2.imwrite(os.getcwd()+'/Masquage et ROI/images/image_roi_modifiee.png', img_resultat)

        clic_coords = None  # Réinitialiser après la modification

    # Quitter le programme en appuyant sur la touche 'O'
    if key == ord('o'):
        break

# Fermer toutes les fenêtres
cv2.destroyAllWindows()
