import cv2
from PIL import Image
from alpr.detect import PlateDetector
from alpr.ocr import PlateReader

video = cv2.VideoCapture("videos/test.mp4")

detector = PlateDetector(
    "best.pt",
    confidence=0.25,
    imgsz=640,
)

reader = PlateReader()

for frame_no in range(1, 101):
    ok, frame = video.read()

    if not ok:
        break

    detections = detector.detect(frame)

    if not detections:
        continue

    # Best plate detection
    detection = detections[0]

    # OpenCV BGR -> PIL RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    result = reader.read(image, detection)

    print(
        f"Frame {frame_no}: "
        f"Plate = '{result.text}' | "
        f"OCR confidence = {result.confidence:.3f}"
    )

video.release()