from ultralytics import YOLO


model = YOLO("/Users/mouhamadou/Desktop/'Videosurveillance algorithmique'/'Fire detec python'/runs/detect/train13/weights/last.pt") 

# Reprendre l'entraînement
results = model.train(
    data="/Users/mouhamadou/Desktop/Innov'athon/datasets/dataset/data.yaml",
    epochs=50,  
    imgsz=640,
    batch=16,
    resume=True 
)