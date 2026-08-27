import cv2
from collections import defaultdict, Counter
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/test.mp4")

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Store vehicle type history
vehicle_history = defaultdict(list)

# Store color history
color_history = defaultdict(list)


def get_color(crop):
    """
    Estimate the dominant vehicle color.
    """

    if crop.size == 0:
        return "unknown"

    h, w = crop.shape[:2]

    if h < 20 or w < 20:
        return "unknown"

    # Remove border/background
    crop = crop[
        int(h * 0.15):int(h * 0.85),
        int(w * 0.15):int(w * 0.85)
    ]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Ignore very dark pixels
    valid = v_channel > 45

    if valid.sum() == 0:
        return "unknown"

    avg_h = h_channel[valid].mean()
    avg_s = s_channel[valid].mean()
    avg_v = v_channel[valid].mean()

    # Black
    if avg_v < 75:
        return "black"

    # White
    if avg_s < 45 and avg_v > 170:
        return "white"

    # Gray
    if avg_s < 55:
        return "gray"

    # Color classification
    if avg_h < 15 or avg_h >= 165:
        return "red"

    elif avg_h < 25:
        return "orange"

    elif avg_h < 35:
        return "yellow"

    elif avg_h < 85:
        return "green"

    elif avg_h < 130:
        return "blue"

    else:
        return "purple"


for frame_no in range(1, 101):

    ok, frame = cap.read()

    if not ok:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        imgsz=640,
        classes=list(vehicle_classes.keys()),
        verbose=False
    )

    current_vehicles = []

    for result in results:

        if result.boxes.id is None:
            continue

        boxes = result.boxes

        track_ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coordinates = boxes.xyxy.int().cpu().tolist()

        for track_id, cls, conf, box in zip(
            track_ids,
            classes,
            confidences,
            coordinates
        ):

            vehicle_type = vehicle_classes[cls]

            x1, y1, x2, y2 = box

            # Keep coordinates inside frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]

            # -------------------------
            # VEHICLE TYPE HISTORY
            # -------------------------

            vehicle_history[track_id].append(vehicle_type)

            vehicle_history[track_id] = \
                vehicle_history[track_id][-20:]

            final_type = Counter(
                vehicle_history[track_id]
            ).most_common(1)[0][0]

            # -------------------------
            # COLOR
            # -------------------------

            color = get_color(crop)

            if color != "unknown":
                color_history[track_id].append(color)

            color_history[track_id] = \
                color_history[track_id][-20:]

            if color_history[track_id]:

                final_color = Counter(
                    color_history[track_id]
                ).most_common(1)[0][0]

            else:
                final_color = "unknown"

            current_vehicles.append(
                (
                    track_id,
                    final_type,
                    final_color,
                    round(conf, 2)
                )
            )

    print(
        f"Frame {frame_no}: {current_vehicles}"
    )


cap.release()

print("\n========== FINAL VEHICLES ==========")

for track_id in vehicle_history:

    final_type = Counter(
        vehicle_history[track_id]
    ).most_common(1)[0][0]

    if color_history[track_id]:

        final_color = Counter(
            color_history[track_id]
        ).most_common(1)[0][0]

    else:

        final_color = "unknown"

    print(
        f"Vehicle ID {track_id}: "
        f"{final_type} | "
        f"{final_color}"
    )