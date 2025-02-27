from ultralytics import YOLO
import cv2


model = YOLO("runs/detect/train13/weights/last.pt")  


video_path = "VideoTest/night.mp4" 
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 2)


    results = model(frame)


    annotated_frame = results[0].plot() 

    cv2.imshow("Analyse en Direct", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()