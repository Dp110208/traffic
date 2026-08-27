# import cv2
# from alpr.detect import PlateDetector

# video = cv2.VideoCapture("videos/test.mp4")

# detector = PlateDetector(
#     "best.pt",
#     confidence=0.25,
#     imgsz=640,
# )

# frame_no = 0

# while frame_no < 100:
#     ok, frame = video.read()

#     if not ok:
#         break

#     frame_no += 1
#     detections = detector.detect(frame)

#     if detections:
#         print(
#             f"Frame {frame_no}: "
#             f"{len(detections)} plate(s), "
#             f"confidence={detections[0].confidence:.3f}"
#         )

# video.release()

# print("Test completed")

# import cv2
# from alpr.detect import PlateDetector

# video = cv2.VideoCapture("videos/test.mp4")

# detector = PlateDetector(
#     "best.pt",
#     confidence=0.10,
#     imgsz=640,
# )

# for frame_no in range(100):
#     ok, frame = video.read()

#     if not ok:
#         break

#     detections = detector.detect(frame)

#     print(f"Frame {frame_no + 1}: {len(detections)} detections")

# video.release()


import cv2
from alpr.detect import PlateDetector

video = cv2.VideoCapture("videos/test.mp4")

detector = PlateDetector(
    "best.pt",
    confidence=0.25,
    imgsz=640,
)

for frame_no in range(1, 101):
    ok, frame = video.read()

    if not ok:
        break

    detections = detector.detect(frame)

    if detections:
        print(
            f"Frame {frame_no}: "
            f"{len(detections)} detections | "
            f"confidences = {[round(d.confidence, 4) for d in detections]}"
        )

video.release()