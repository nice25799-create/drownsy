import cv2
import numpy as np

def try_open(cam_idx, backend):
    """Try to open a camera; return the VideoCapture object or None."""
    cap = cv2.VideoCapture(cam_idx, backend)

    if not cap.isOpened():
        cap.release()
        return None

    return cap

def main():
    # Common backends on Windows:
    # CAP_DSHOW = DirectShow
    # CAP_MSMF  = Media Foundation
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (0, "Default")  # Let OpenCV choose
    ]

    opened = False

    for backend, name in backends:
        for idx in range(0, 3):  # Try camera indices 0,1,2
            print(f"Trying camera {idx} with backend {name} ...")

            cap = try_open(idx, backend)

            if cap is not None:
                print(f"  ✅ Opened camera {idx} via {name}")
                opened = True
                break
            else:
                print("  ❌ Failed to open")

        if opened:
            break

    if not opened:
        print("❌ Could not open any webcam. Check:")
        print("   - Is the camera plugged in?")
        print("   - Is another app using it? (Close Zoom/Teams/etc.)")
        print("   - Do you have the correct drivers?")
        return

    # Try to force a stable format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Try MJPEG format
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*'MJPG')
    )

    print("\n📹 Press 'q' in the video window to quit.\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠️ Empty frame received – breaking.")
            break

        frame_count += 1

        if frame_count == 1:
            # Print info only once
            print(f"Raw frame  : shape={frame.shape}, dtype={frame.dtype}")

            # Sample pixel value
            print(f"Sample pixel (top-left): {frame[0,0]}")

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Ensure correct memory layout for dlib/OpenCV
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        if frame_count == 1:
            print(
                f"Gray frame : "
                f"shape={gray.shape}, "
                f"dtype={gray.dtype}, "
                f"contiguous={gray.flags['C_CONTIGUOUS']}"
            )

        # Show frames
        cv2.imshow("Raw BGR frame", frame)
        cv2.imshow("Gray frame", gray)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("✅ Test finished.")

if __name__ == "__main__":
    main()