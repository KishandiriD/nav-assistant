import cv2, time, threading
from collections import deque
from ultralytics import YOLO
import pyttsx3

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

history = {}
speaking = False
last_spoken = 0

def speak(text):
    global speaking
    if speaking:
        return
    speaking = True
    def run():
        global speaking
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        speaking = False
    threading.Thread(target=run, daemon=True).start()

def get_zone(xc, w):
    if xc < w / 3: return "left"
    if xc < 2 * w / 3: return "center"
    return "right"

def get_distance(ratio):
    if ratio > 0.6: return "very near"
    if ratio > 0.35: return "near"
    return "far"

def is_approaching(name, ratio):
    h = history.setdefault(name, deque(maxlen=8))
    h.append(ratio)
    return len(h) == 8 and (h[-1] - h[0]) > 0.08

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    result = model(frame, verbose=False)[0]

    seen, occupied, danger = set(), set(), None

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = model.names[int(box.cls[0])]
        ratio = (y2 - y1) / h
        zone = get_zone((x1 + x2) / 2, w)
        dist = get_distance(ratio)
        appr = is_approaching(name, ratio)
        seen.add(name)

        if dist != "far":
            occupied.add(zone)

        risky = (zone == "center" and dist != "far") or appr
        color = (0, 0, 255) if risky else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} | {zone} | {dist}" + (" | approaching" if appr else "")
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if risky and (danger is None or ratio > danger[3]):
            danger = (name, zone, appr, ratio)

    for k in list(history):
        if k not in seen:
            del history[k]

    if danger and time.time() - last_spoken > 3:
        name, zone, appr, _ = danger
        if "left" not in occupied:
            action = "move left"
        elif "right" not in occupied:
            action = "move right"
        else:
            action = "stop"
        msg = f"{name} {zone}" + (", approaching" if appr else "") + f". {action}"
        speak(msg)
        last_spoken = time.time()

    cv2.line(frame, (w // 3, 0), (w // 3, h), (255, 255, 255), 1)
    cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 1)
    cv2.imshow("Nav", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()