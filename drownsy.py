import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import os
import threading
from collections import deque
from statistics import median

import anthropic

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

# -------------------- CONFIGURATION --------------------
EYE_CLOSED_ALERT_SEC = 1.0    # continuous eye closure that fires the alert

MOUTH_AR_THRESH = 0.5         # MAR above this → mouth wide open (yawning)
MOUTH_OPEN_YAWN_SEC = 0.5     # continuous open mouth that counts as one yawn

CALIB_SECONDS = 10.0          # startup calibration capture duration
CALIB_RATIO = 0.75            # threshold = ratio x median open-eye EAR
CALIB_THRESH_MIN = 0.15       # clamp for the derived threshold
CALIB_THRESH_MAX = 0.30
MIN_CALIB_SAMPLES = 30        # fewer face-frames than this -> use default
EAR_DEFAULT_THRESH = 0.25     # fallback when calibration fails

PERCLOS_WINDOW_SEC = 60.0     # rolling window for PERCLOS
PERCLOS_THRESH = 0.30         # PERCLOS value that fires an alert

YAWN_WINDOW_SEC = 60.0        # rolling window for yawn frequency
YAWN_ALERT_COUNT = 3          # yawns within the window to trigger an alert

CLAUDE_MODEL = "claude-opus-4-7"  # swap to "claude-haiku-4-5" for faster/cheaper alerts
ALERT_COOLDOWN = 6.0              # min seconds between spoken alerts
# -------------------------------------------------------

def eye_aspect_ratio(eye):
    """Compute the Eye Aspect Ratio (EAR) for a single eye."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    """Compute the Mouth Aspect Ratio (MAR) from the 8 inner-lip landmarks."""
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    D = dist.euclidean(mouth[0], mouth[4])

    return (A + B + C) / (2.0 * D)

def derive_threshold(samples):
    """Personal EAR threshold from calibration samples: ratio of the median."""
    if len(samples) < MIN_CALIB_SAMPLES:
        return EAR_DEFAULT_THRESH, True

    open_ear = median(samples)
    threshold = CALIB_RATIO * open_ear
    threshold = max(CALIB_THRESH_MIN, min(CALIB_THRESH_MAX, threshold))

    return threshold, False


class PerclosTracker:
    """Rolling-window PERCLOS: fraction of recent frames with eyes closed."""

    def __init__(self, window_sec):
        self.window_sec = window_sec
        self.samples = deque()   # (timestamp, closed)

    def update(self, closed, t):
        self.samples.append((t, closed))

        while self.samples and (t - self.samples[0][0]) > self.window_sec:
            self.samples.popleft()

    def value(self):
        if not self.samples:
            return 0.0

        closed_count = sum(1 for _, closed in self.samples if closed)
        return closed_count / len(self.samples)

    def ready(self):
        if len(self.samples) < 2:
            return False

        span = self.samples[-1][0] - self.samples[0][0]
        return span >= 0.5 * self.window_sec

    def clear(self):
        self.samples.clear()


def calibrate_ear(vs, detector, predictor, l_idx, r_idx):
    """Collect open-eye EAR samples for CALIB_SECONDS, then derive a threshold.

    Returns (threshold, used_default), or None if the user pressed q.
    """
    (lStart, lEnd) = l_idx
    (rStart, rEnd) = r_idx

    samples = []
    start = time.time()

    while True:
        remaining = CALIB_SECONDS - (time.time() - start)

        if remaining <= 0:
            break

        ret, frame = vs.read()

        if not ret:
            print("[ERROR] Failed to read frame during calibration.")
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        ear = None

        if rects:
            shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)

            leftEAR = eye_aspect_ratio(shape[lStart:lEnd])
            rightEAR = eye_aspect_ratio(shape[rStart:rEnd])
            ear = (leftEAR + rightEAR) / 2.0
            samples.append(ear)

        cv2.putText(
            frame,
            f"CALIBRATING - look at camera, eyes open ({remaining:.0f}s)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2
        )

        if ear is not None:
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2
            )

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    threshold, used_default = derive_threshold(samples)

    if used_default:
        print(f"[WARN] Calibration failed ({len(samples)} samples); "
              f"using default EAR threshold {threshold:.2f}.")
    else:
        print(f"[INFO] Calibrated EAR threshold: {threshold:.3f} "
              f"(from {len(samples)} samples)")

    return threshold, used_default


def generate_alert_message(client, episode_count, reason):
    """Ask Claude for a short, context-aware wake-up line. Falls back on failure."""
    fallback = "Wake up! You are showing signs of drowsiness."

    if client is None:
        return fallback

    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=60,
            system=(
                "You generate ONE very short spoken wake-up line for a drowsy "
                "driver-alert system. Max 12 words, urgent but calm, and vary the "
                "wording each time. Output only the sentence: no quotes, no emojis, "
                "no preamble."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Drowsiness detected. This is alert #{episode_count} this "
                    f"session. {reason} Give the wake-up line."
                ),
            }],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"[WARN] Claude alert generation failed: {e}")
        return fallback

def speak(text):
    """Print and (if available) speak the alert message aloud."""
    print(f"[ALERT] {text}")

    if not _TTS_AVAILABLE:
        return

    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[WARN] Text-to-speech failed: {e}")

def handle_alert(client, episode_count, reason, busy_flag):
    """Background worker: generate the message, then speak it."""
    try:
        message = generate_alert_message(client, episode_count, reason)
        speak(message)
    finally:
        busy_flag.clear()   # allow the next alert

def main():
    print("[INFO] Loading facial landmark predictor...")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat"
    )

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = (60, 68)   # inner-lip landmarks (dlib points 61-68)

    # Claude client — reads ANTHROPIC_API_KEY from the environment.
    try:
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"[WARN] Claude client unavailable ({e}); using fallback alerts.")
        client = None

    print("[INFO] Starting video stream...")

    vs = cv2.VideoCapture(0)   # 0 = default webcam
    time.sleep(2.0)            # let camera warm up

    calib = calibrate_ear(
        vs, detector, predictor, (lStart, lEnd), (rStart, rEnd)
    )

    if calib is None:
        print("[INFO] Calibration aborted; exiting.")
        vs.release()
        cv2.destroyAllWindows()
        return

    ear_thresh, ear_thresh_is_default = calib

    closure_start = None        # when the current eye-closure began
    ALARM_ON = False

    mouth_open_start = None     # when the current mouth-open episode began
    yawn_counted = False        # current mouth-open episode already counted
    yawn_times = deque()        # timestamps of recent yawns
    episode_count = 0           # how many alerts this session
    last_alert_time = 0.0       # for the cooldown
    alert_busy = threading.Event()   # set while an alert thread is running
    perclos_tracker = PerclosTracker(PERCLOS_WINDOW_SEC)

    while True:
        ret, frame = vs.read()

        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # Resize for speed
        frame = imutils.resize(frame, width=450)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        rects = detector(gray, 0)

        for rect in rects:
            # Facial landmarks
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Compute EAR
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            # Draw contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(
                frame,
                [leftEyeHull],
                -1,
                (0, 255, 0),
                1
            )

            cv2.drawContours(
                frame,
                [rightEyeHull],
                -1,
                (0, 255, 0),
                1
            )

            # Extract inner-lip coordinates and compute MAR
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)

            mouthHull = cv2.convexHull(mouth)

            cv2.drawContours(
                frame,
                [mouthHull],
                -1,
                (0, 255, 255),
                1
            )

            # Yawn detection: mouth wide open for enough consecutive time
            now = time.time()
            perclos_tracker.update(ear < ear_thresh, now)
            perclos = perclos_tracker.value()

            if mar > MOUTH_AR_THRESH:
                if mouth_open_start is None:
                    mouth_open_start = now

                if (now - mouth_open_start) >= MOUTH_OPEN_YAWN_SEC:
                    # Count each open-mouth episode as one yawn
                    if not yawn_counted:
                        yawn_counted = True
                        yawn_times.append(now)

                    cv2.putText(
                        frame,
                        "YAWNING",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
            else:
                mouth_open_start = None
                yawn_counted = False

            # Drop yawns that fell out of the rolling window
            while yawn_times and (now - yawn_times[0]) > YAWN_WINDOW_SEC:
                yawn_times.popleft()

            # Too many yawns in the window → spoken alert
            if (len(yawn_times) >= YAWN_ALERT_COUNT
                    and not alert_busy.is_set()
                    and (now - last_alert_time) > ALERT_COOLDOWN):
                last_alert_time = now
                episode_count += 1
                reason = (f"Driver yawned {len(yawn_times)} times within "
                          "the last minute.")
                yawn_times.clear()   # start a fresh window after alerting

                alert_busy.set()
                threading.Thread(
                    target=handle_alert,
                    args=(client, episode_count, reason, alert_busy),
                    daemon=True,
                ).start()

            # Sustained partial closure over the window -> spoken alert
            if (perclos_tracker.ready()
                    and perclos >= PERCLOS_THRESH
                    and not alert_busy.is_set()
                    and (now - last_alert_time) > ALERT_COOLDOWN):
                last_alert_time = now
                episode_count += 1
                reason = (f"Driver's eyes were closed "
                          f"{perclos:.0%} of the last minute.")
                perclos_tracker.clear()   # restart the window after alerting

                alert_busy.set()
                threading.Thread(
                    target=handle_alert,
                    args=(client, episode_count, reason, alert_busy),
                    daemon=True,
                ).start()

            # Drowsiness detection
            if ear < ear_thresh:
                if closure_start is None:
                    closure_start = now   # episode begins

                closed_dur = now - closure_start

                if closed_dur >= EYE_CLOSED_ALERT_SEC:
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    # Fire one spoken alert per episode, respecting the cooldown,
                    # and only if no alert is already being spoken.
                    if (not ALARM_ON
                            and not alert_busy.is_set()
                            and (now - last_alert_time) > ALERT_COOLDOWN):
                        ALARM_ON = True
                        last_alert_time = now
                        episode_count += 1
                        reason = (f"Eyes were closed for about "
                                  f"{closed_dur:.0f} seconds.")

                        alert_busy.set()
                        threading.Thread(
                            target=handle_alert,
                            args=(client, episode_count, reason, alert_busy),
                            daemon=True,
                        ).start()

            else:
                closure_start = None
                ALARM_ON = False

            # Display EAR / MAR values and yawn count
            thresh_note = " default" if ear_thresh_is_default else ""

            cv2.putText(
                frame,
                f"EAR: {ear:.2f} (th {ear_thresh:.2f}{thresh_note})",
                (230, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (300, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            perclos_color = (
                (0, 0, 255) if perclos >= PERCLOS_THRESH else (0, 255, 255)
            )

            cv2.putText(
                frame,
                f"PERCLOS: {perclos:.0%}",
                (300, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                perclos_color,
                2
            )

            cv2.putText(
                frame,
                f"Yawns(60s): {len(yawn_times)}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # Show frame
        cv2.imshow("Frame", frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    print("[INFO] Releasing resources...")

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()