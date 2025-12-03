
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import time

# ----- CONFIG -----
YOLO_MODEL = "license_plate_detector.pt"
CONF_THRESHOLD = 0.35
OCR_LANGS = ['en']
PLATE_REGEX = r'[A-Z0-9]{4,8}'
VIDEO_SOURCE = 0
RUN_EVERY_N_FRAMES = 10
SHOW = True
# ------------------

# 🔥 ATIVAR GPU NO YOLO
detector = YOLO(YOLO_MODEL)
detector.to('cuda')

# 🔥 ATIVAR GPU NO EASYOCR
reader = easyocr.Reader(OCR_LANGS, gpu=True)

def postprocess_text(txt):
    if txt is None:
        return ""
    s = re.sub(r'[^A-Z0-9]', '', txt.upper())
    m = re.search(PLATE_REGEX, s)
    return m.group(0) if m else s


def detect_only(frame):
    results = detector(frame, conf=CONF_THRESHOLD, verbose=False)
    boxes_detected = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes_detected.append((x1, y1, x2, y2, conf))
    return boxes_detected


def read_plate(crop):
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ocr_results = reader.readtext(gray, detail=0)
        if len(ocr_results) > 0:
            txt = max(ocr_results, key=len)
            return postprocess_text(txt)
        return ""
    except:
        return ""


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    cap.set(3, 640)
    cap.set(4, 480)

    frame_count = 0
    last_saved = 0
    boxes_cache = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % RUN_EVERY_N_FRAMES == 0:
            boxes_cache = detect_only(frame)

        for (x1, y1, x2, y2, conf) in boxes_cache:
            crop = frame[y1:y2, x1:x2]
            txt = read_plate(crop)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, txt, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cur = time.time()
            if txt and cur - last_saved > 1:
                filename = f"plates/{int(cur)}_{txt}.jpg"
                cv2.imwrite(filename, crop)
                last_saved = cur

        if SHOW:
            cv2.imshow("SPIA - Plate Detection (GPU)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
