from ultralytics import YOLO
import cv2
import cvzone

# ----------------------------
# Intel Mac (CPU) optimized settings
# ----------------------------
VIDEO_PATH = "../videos/vehicles.mp4"
MODEL_PATH = "../YOLO-Weights/yolov8n.pt"  # fastest on CPU

CONF_THRES = 0.15
IOU_THRES = 0.50
IMGSZ = 640

FRAME_SKIP = 2          # run YOLO every 2nd frame (set 3 for more speed)
RESIZE_WIDTH = 1280     # downscale frame for speed; set None to disable

# COCO class ids (Ultralytics COCO): car=2, motorbike=3, bus=5, truck=7
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_CLASSES = {"car", "motorbike", "bus", "truck"}

# Horizontal counting line Y (in ORIGINAL video coordinates before resizing)
# Your previous code used y=340
LINE_Y = 340

COUNT_BOTH_DIRECTIONS = True  # False => only top->bottom

# ----------------------------
# Setup
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

model = YOLO(MODEL_PATH)

# Optional badge overlay
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
if imgGraphics is None:
    print("Warning: graphics.png not found. Running without badge.")
else:
    # make it small
    target_w = 220
    scale = target_w / imgGraphics.shape[1]
    target_h = int(imgGraphics.shape[0] * scale)
    imgGraphics = cv2.resize(imgGraphics, (target_w, target_h))

    # ensure alpha
    if imgGraphics.shape[2] == 3:
        imgGraphics = cv2.cvtColor(imgGraphics, cv2.COLOR_BGR2BGRA)

badge_x, badge_y = 20, 20

# Counting state
counted_ids = set()
last_cy = {}

frame_idx = 0
last_results = None

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Resize frame for speed
    scale_factor = 1.0
    if RESIZE_WIDTH is not None and img.shape[1] > RESIZE_WIDTH:
        scale_factor = RESIZE_WIDTH / img.shape[1]
        img = cv2.resize(img, (RESIZE_WIDTH, int(img.shape[0] * scale_factor)))

    # FULL-WIDTH horizontal line (scaled)
    scaled_line_y = int(LINE_Y * scale_factor)
    x1_line, x2_line = 0, img.shape[1]
    limits = [x1_line, scaled_line_y, x2_line, scaled_line_y]

    # Badge overlay
    if imgGraphics is not None:
        img = cvzone.overlayPNG(img, imgGraphics, (badge_x, badge_y))

    # Run YOLO tracking only every N frames for speed
    if frame_idx % FRAME_SKIP == 0:
        last_results = model.track(
            img,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMGSZ,
            device="cpu",
            verbose=False,
            classes=VEHICLE_CLASS_IDS,
            max_det=50,
        )

    # Draw counting line (full width)
    cv2.line(
        img,
        (limits[0], limits[1]),
        (limits[2], limits[3]),
        (0, 0, 255),
        4,
    )

    # Draw boxes + count using last results
    if last_results is not None:
        r = last_results[0]
        boxes = r.boxes

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            for box, tid in zip(boxes, boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                tid = int(tid.item())

                cls = int(box.cls[0])
                name = r.names.get(cls, str(cls))
                if name not in VEHICLE_CLASSES:
                    continue

                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                cvzone.cornerRect(img, (x1, y1, w, h), l=7, rt=2, colorR=(255, 0, 0))
                cvzone.putTextRect(
                    img,
                    f"{name} ID:{tid}",
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1,
                    offset=3,
                )

                # Count by crossing the horizontal line (FULL WIDTH: no x-range restriction)
                prev = last_cy.get(tid)
                last_cy[tid] = cy
                if prev is None:
                    continue

                crossed_down = prev < scaled_line_y and cy >= scaled_line_y
                crossed_up = prev > scaled_line_y and cy <= scaled_line_y
                crossed = (crossed_down or crossed_up) if COUNT_BOTH_DIRECTIONS else crossed_down

                if crossed and tid not in counted_ids:
                    counted_ids.add(tid)
                    cv2.line(
                        img,
                        (limits[0], limits[1]),
                        (limits[2], limits[3]),
                        (0, 255, 0),
                        4,
                    )

    # Count display next to badge
    count_x = badge_x + (imgGraphics.shape[1] if imgGraphics is not None else 0) + 15
    count_y = badge_y + 60
    cv2.putText(
        img,
        str(len(counted_ids)),
        (count_x, count_y),
        cv2.FONT_HERSHEY_PLAIN,
        4,
        (50, 50, 255),
        6,
    )

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()