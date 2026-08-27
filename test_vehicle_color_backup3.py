# import cv2
# from collections import Counter
# from ultralytics import YOLO

# model = YOLO("yolov8n.pt")

# cap = cv2.VideoCapture("videos/test.mp4")

# vehicle_classes = {
#     2: "car",
#     3: "motorcycle",
#     5: "bus",
#     7: "truck",
# }

# # Store predictions for each rough vehicle position
# vehicle_history = []


# def find_matching_vehicle(center_x, center_y):
#     """
#     Find an existing vehicle whose center is close
#     to the current detection.
#     """
#     best_match = None
#     best_distance = 100

#     for vehicle in vehicle_history:
#         dx = center_x - vehicle["center_x"]
#         dy = center_y - vehicle["center_y"]

#         distance = (dx * dx + dy * dy) ** 0.5

#         if distance < best_distance:
#             best_distance = distance
#             best_match = vehicle

#     return best_match


# for frame_no in range(1, 101):

#     ok, frame = cap.read()

#     if not ok:
#         break

#     results = model(
#         frame,
#         conf=0.35,
#         imgsz=640,
#         verbose=False
#     )

#     current_vehicles = []

#     for result in results:

#         for box in result.boxes:

#             cls = int(box.cls[0])
#             conf = float(box.conf[0])

#             if cls not in vehicle_classes:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2

#             vehicle_type = vehicle_classes[cls]

#             # Try to match this vehicle with previous frames
#             matched = find_matching_vehicle(center_x, center_y)

#             if matched is None:

#                 matched = {
#                     "center_x": center_x,
#                     "center_y": center_y,
#                     "history": []
#                 }

#                 vehicle_history.append(matched)

#             matched["center_x"] = center_x
#             matched["center_y"] = center_y

#             matched["history"].append(vehicle_type)

#             # Keep only last 15 predictions
#             matched["history"] = matched["history"][-15:]

#             # Majority voting
#             final_type = Counter(
#                 matched["history"]
#             ).most_common(1)[0][0]

#             current_vehicles.append(
#                 (
#                     final_type,
#                     round(conf, 2),
#                     len(matched["history"])
#                 )
#             )

#     print(
#         f"Frame {frame_no}: {current_vehicles}"
#     )


# cap.release()


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

# Store vehicle type history by tracking ID
vehicle_history = defaultdict(list)

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

        for track_id, cls, conf in zip(
            track_ids,
            classes,
            confidences
        ):

            vehicle_type = vehicle_classes[cls]

            # Save prediction for this specific vehicle ID
            vehicle_history[track_id].append(vehicle_type)

            # Keep last 20 predictions
            vehicle_history[track_id] = \
                vehicle_history[track_id][-20:]

            # Majority vote
            final_type = Counter(
                vehicle_history[track_id]
            ).most_common(1)[0][0]

            current_vehicles.append(
                (
                    track_id,
                    final_type,
                    round(conf, 2)
                )
            )

    print(
        f"Frame {frame_no}: {current_vehicles}"
    )

cap.release()

print("\n========== FINAL VEHICLES ==========")

for track_id, history in vehicle_history.items():

    final_type = Counter(history).most_common(1)[0][0]

    print(
        f"Vehicle ID {track_id}: "
        f"{final_type} "
        f"({len(history)} observations)"
    )