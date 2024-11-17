from ultralytics import YOLO
model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)
results = model.train(data="FLIR.yaml", epochs=100, imgsz=640, save=True, patience=0, batch=0.90, save_period=1, pretrained=True, freeze=None, plots=True)