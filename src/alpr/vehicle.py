from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.35,
    ):
        self.model = YOLO(weights)
        self.confidence = confidence

        self.type_history = defaultdict(list)
        self.color_history = defaultdict(list)

    def get_color(self, crop):
        if crop.size == 0:
            return "unknown"

        h, w = crop.shape[:2]

        if h < 20 or w < 20:
            return "unknown"

        crop = crop[
            int(h * 0.15):int(h * 0.85),
            int(w * 0.15):int(w * 0.85)
        ]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        valid = v_channel > 45

        if valid.sum() == 0:
            return "unknown"

        avg_h = h_channel[valid].mean()
        avg_s = s_channel[valid].mean()
        avg_v = v_channel[valid].mean()

        if avg_v < 75:
            return "black"

        if avg_s < 45 and avg_v > 170:
            return "white"

        if avg_s < 55:
            return "gray"

        if avg_h < 10 or avg_h >= 170:
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
            return "red"

    def detect(self, frame) -> list[dict[str, Any]]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            imgsz=640,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        vehicles = []

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
                coordinates,
            ):
                vehicle_type = VEHICLE_CLASSES[cls]

                x1, y1, x2, y2 = box

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                self.type_history[track_id].append(vehicle_type)
                self.type_history[track_id] = self.type_history[track_id][-20:]

                final_type = Counter(
                    self.type_history[track_id]
                ).most_common(1)[0][0]

                color = self.get_color(crop)

                if color != "unknown":
                    self.color_history[track_id].append(color)

                self.color_history[track_id] = self.color_history[track_id][-20:]

                if self.color_history[track_id]:
                    final_color = Counter(
                        self.color_history[track_id]
                    ).most_common(1)[0][0]
                else:
                    final_color = "unknown"

                vehicles.append({
                    "track_id": track_id,
                    "type": final_type,
                    "color": final_color,
                    "confidence": round(conf, 2),
                    "box": (x1, y1, x2, y2),
                })

        return vehicles