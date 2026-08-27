import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("videos/test.mp4")

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

for frame_no in range(1, 101):
    ok, frame = video.read()

    if not ok:
        break

    results = model.predict(
        frame,
        conf=0.4,
        imgsz=640,
        verbose=False,
    )

    vehicles = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in vehicle_classes:
            vehicles.append(
                (vehicle_classes[cls], round(conf, 2))
            )

    if vehicles:
        print(f"Frame {frame_no}: {vehicles}")

video.release()