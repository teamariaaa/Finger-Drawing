import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

# ---------------- Tunables ----------------
PINCH_THRESHOLD = 0.05     # pinch sensitivity (lower = tighter pinch needed)
LINE_THICKNESS = 6         # user stroke thickness
SMOOTHING_ALPHA = 0.35     # fingertip smoothing (0..1)
FIXED_TOL_PX = 10          # fixed tolerance for scoring (not shown/adjustable)
TARGET_PATH = "contour.png"  # optional: white line on black/transparent background

WINDOW_NAME = "Trace the Contour - AI Drawing"

# --------------- Helpers ------------------
def norm_dist(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return math.hypot(dx, dy)

def to_px(norm_xy, w, h):
    return (int(norm_xy[0] * w), int(norm_xy[1] * h))

def make_circle_mask(shape, radius_ratio=0.35, thickness=6):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(w, h) * radius_ratio)
    cv2.circle(mask, center, radius, 255, thickness, lineType=cv2.LINE_AA)
    return mask

def load_target_mask(shape, path):
    h, w = shape[:2]
    if not os.path.exists(path):
        return make_circle_mask(shape)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return make_circle_mask(shape)

    if img.ndim == 2:  # grayscale
        mask = img
    elif img.shape[2] == 4:  # RGBA
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        mask = np.maximum(alpha, gray)
    else:  # BGR
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask

def make_binary_strokes(canvas_bgr):
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    _, binmask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return binmask

def score_f1(target_mask, user_mask, tol_px):
    if tol_px < 0:
        tol_px = 0
    if np.count_nonzero(target_mask) == 0 and np.count_nonzero(user_mask) == 0:
        return 100.0, 100.0, 100.0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(1, 2 * tol_px + 1), max(1, 2 * tol_px + 1))
    )

    target_dil = cv2.dilate(target_mask, kernel)
    user_and_target = cv2.bitwise_and(user_mask, target_dil)
    user_px = np.count_nonzero(user_mask)
    precision = (np.count_nonzero(user_and_target) / user_px * 100) if user_px > 0 else 0.0

    user_dil = cv2.dilate(user_mask, kernel)
    target_and_user = cv2.bitwise_and(target_mask, user_dil)
    target_px = np.count_nonzero(target_mask)
    recall = (np.count_nonzero(target_and_user) / target_px * 100) if target_px > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * (precision * recall) / (precision + recall))
    return precision, recall, f1

def overlay_target(frame, target_mask, color=(255, 0, 0)):
    overlay = frame.copy()
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, color, 2, lineType=cv2.LINE_AA)
    return overlay

def put_text(img, text, org, scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)

# --------------- Main App -----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 60)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

canvas = None
prev_idx_tip = None
prev_pt = None
ema_pt = None
show_target = True
target_mask = None
fullscreen = False
drawing_now = False
fingertip_px = None
pinch_val = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)
        target_mask = load_target_mask(frame.shape, TARGET_PATH)

    
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:

        lm = result.multi_hand_landmarks[0].landmark

        print(drawing_now, prev_idx_tip)
        if drawing_now and prev_idx_tip is not None:
            chosen_hand = min(
                result.multi_hand_landmarks,
                key=lambda hand: (
                    (hand.landmark[8].x - prev_idx_tip[0])**2 +
                    (hand.landmark[8].y - prev_idx_tip[1])**2
                )
            )
            lm = chosen_hand.landmark
            
        idx_tip = (lm[8].x, lm[8].y)
        prev_idx_tip = idx_tip
        thumb_tip = (lm[4].x, lm[4].y)
        pinch_val = norm_dist(idx_tip, thumb_tip)
        fingertip_px = to_px(idx_tip, w, h)

        if ema_pt is None:
            ema_pt = fingertip_px
        else:
            ema_pt = (
                int(SMOOTHING_ALPHA * fingertip_px[0] + (1 - SMOOTHING_ALPHA) * ema_pt[0]),
                int(SMOOTHING_ALPHA * fingertip_px[1] + (1 - SMOOTHING_ALPHA) * ema_pt[1])
            )

        drawing_now = pinch_val < PINCH_THRESHOLD
        cv2.circle(frame, fingertip_px, 10, (0, 0, 255), -1)

    if drawing_now and ema_pt is not None:
        if prev_pt is not None:
            cv2.line(canvas, prev_pt, ema_pt, (0, 255, 0),
                     LINE_THICKNESS, lineType=cv2.LINE_AA)
        prev_pt = ema_pt
    else:
        prev_pt = None

    user_mask = make_binary_strokes(canvas)
    precision, recall, f1 = score_f1(target_mask, user_mask, FIXED_TOL_PX)

    display = frame.copy()
    if show_target:
        display = overlay_target(display, target_mask, color=(255, 128, 0))
    combined = cv2.addWeighted(display, 0.7, canvas, 0.9, 0)

    # --- HUD (short, fits screen) ---

    y0 = 30
    if pinch_val is not None:
        y0 += 30
    put_text(combined,
             f"F1: {f1:5.1f}%",
             (10, y0))
    y0 += 30

    cv2.imshow(WINDOW_NAME, combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('s'):
        fname = f"drawing_{int(time.time())}.png"
        cv2.imwrite(fname, canvas)
        print("Saved:", fname)
    elif key == ord('t'):
        show_target = not show_target
    elif key == ord('f'):  # fullscreen toggle
        fullscreen = not fullscreen
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        )

cap.release()
cv2.destroyAllWindows()
