from ultralytics import YOLO

model = YOLO("best.pt")

results = model(
    "videos/test.mp4",
    stream=True,
    max_det=20,
    conf=0.25,
    vid_stride=5,
    save=True
)

count = 0

for result in results:
    count += 1

    if len(result.boxes) > 0:
        print(f"Frame {count}: {len(result.boxes)} plate(s) detected")

    if count >= 100:
        break

print("Test completed")