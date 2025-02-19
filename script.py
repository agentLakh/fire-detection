from ultralytics import YOLO
import cv2

# Charger le modèle YOLOv8
model = YOLO("runs/detect/train13/weights/best.pt")  # Remplacez par le chemin de votre fichier .pt

# Charger la vidéo
video_path = "VideoTest/night.mp4"  # Remplacez par le chemin de votre vidéo
cap = cv2.VideoCapture(0)

# Vérifier si la vidéo est ouverte correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Boucle principale pour lire et afficher la vidéo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Sauter des frames pour accélérer le traitement
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 2)

    # Détecter les objets dans la frame avec YOLOv8
    results = model(frame)

    # Afficher les résultats
    annotated_frame = results[0].plot()  # Dessiner les boîtes englobantes et les étiquettes

    # Afficher la frame analysée dans une fenêtre
    cv2.imshow("Analyse en Direct", annotated_frame)
    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()