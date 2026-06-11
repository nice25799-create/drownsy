---
name: camtest
description: Test the webcam before running the drowsiness detector. Runs camtest.py, which tries camera indices 0-2 across DirectShow/MSMF/default backends and opens a preview window.
---

Run the webcam diagnostic for this project:

1. Run `python camtest.py` from the project root (D:\project). It opens preview windows; the user closes them by pressing `q` in the video window.
2. The script prints which camera index + backend worked, the frame shape/dtype, and whether the grayscale frame is C-contiguous (required by dlib).
3. If no camera opens, walk the user through the printed checklist: camera plugged in, not in use by another app (Zoom/Teams), drivers OK. Also suggest checking Windows camera privacy settings (Settings > Privacy & security > Camera) if all backends fail.
4. If a camera works but only at a non-zero index or non-default backend, point out that drownsy.py uses `cv2.VideoCapture(0)` and offer to update it to match the working index/backend.

This is an interactive GUI script — run it in the foreground and let the user interact with the window.
