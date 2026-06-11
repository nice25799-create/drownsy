# Drowsiness Detection — Project Context & Research References

> Drop this file into your project root (or merge into `CLAUDE.md`) so the coding
> assistant has full context on what is being built, the standard methods and
> metrics, the public datasets, and the literature.

---

## 1. What this project is

A **drowsiness detection** system. The primary research angle of interest is
**drowsiness and its relationship to posture** (head pose, body/sitting posture,
seat-pressure), in addition to the more common eye/face-based cues. The system
should detect early signs of fatigue/sleepiness and raise an alert.

Typical pipeline:

1. **Capture** — webcam / IR camera / dashboard camera (optionally physiological
   or vehicle sensors).
2. **Detect & track** — locate face/eyes/mouth (Dlib 68-landmarks, Haar cascade,
   MTCNN, MediaPipe, or YOLO for ROI) and/or body joints (OpenPose, BlazePose,
   MediaPipe Pose).
3. **Extract features** — EAR, MAR, PERCLOS, blink rate, yawn frequency, head
   pose angles (pitch/yaw/roll), nodding, posture/pressure indices.
4. **Classify** — threshold rules, classic ML (SVM, KNN, Random Forest), or deep
   learning (CNN, CNN-LSTM, 3D-CNN, Vision Transformers, YOLO).
5. **Alert** — alarm, dashboard warning, IoT/cloud notification, rest-stop
   suggestion.

---

## 2. Key concepts & metrics (vocabulary the code should use)

- **EAR (Eye Aspect Ratio)** — ratio of vertical to horizontal eye-landmark
  distances; drops toward 0 when the eye closes. A common drowsy threshold is
  ~0.2–0.25 sustained over N frames.
- **MAR (Mouth Aspect Ratio)** — analogous ratio for the mouth; rises during a
  yawn.
- **PERCLOS** — *Percentage of Eye Closure* over time; the most widely used
  vision-based drowsiness measure. Often flagged when PERCLOS ≥ 0.3 (or higher).
- **Head pose** — pitch/yaw/roll of the head; nodding, tilting and gradual
  slumping often precede obvious fatigue.
- **Posture / seat-pressure indices** — Center-of-Pressure (COP) movement and
  fractal-dimension of back/seat pressure change with arousal level; posture also
  estimable from upper-body joints (shoulders, neck).
- **HRV (Heart Rate Variability)** — derived from ECG/PPG R–R intervals; changes
  with drowsiness.
- **Modalities** are usually grouped as: **behavioral** (face/eye/mouth/head/body),
  **vehicular** (steering angle, lane departure, braking), and **physiological**
  (EEG, ECG, PPG, EMG, EOG, respiration).

---

## 3. Public datasets

| Dataset | Subjects | Signals / Modality | Notes |
|---|---|---|---|
| **NTHU-DDD** | 36 | IR video, 5 scenarios | Bare face / glasses / sunglasses, day & night; ~9.5 h. Most-cited benchmark. |
| **UTA-RLDD (RLDD)** | 60 | ~30 h RGB video, 3 classes | Real-life drowsiness, multi-stage labels. |
| **YawDD** | ~107 | Video, mouth states | Yawning-focused; two camera placements. |
| **DROZY** | 14 | EEG, EOG, EKG, EMG, NIR | Rich physiological multimodal set. |
| **DMD** | 37 | Video (face/body/hand) | Driver Monitoring Dataset. |
| **3MDAD** | 50 | RGB + IR + depth | Multimodal multiview. |
| **MRL Eye** | 37 | 15,000 IR eye images | Eye-state classification. |
| **NITYMED** | 21 | Video, 5 classes | — |
| **SEED-VIG** | 23 | EEG + EOG | Vigilance estimation. |
| **UL-DD** | 19 | RGB/IR/pose, grip, telemetry, biometrics | Newest, very rich multimodal. |

- **NTHU-DDD request page:** http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
- **UL-DD (paper + dataset description):** https://arxiv.org/pdf/2507.13403

---

## 4. Reference library

> Links prioritize **free, downloadable PDFs**. arXiv, MDPI, PMC, Frontiers,
> Nature, IET (Wiley OA), and direct `.pdf` links are open. ResearchGate /
> Academia.edu links usually need a free account; ask for an open mirror if a
> specific one is gated.

### 4.1 Surveys & reviews (start here for the literature review)
- A Survey on Drowsiness Detection — Modern Applications and Methods (2024) — https://arxiv.org/pdf/2408.12990
- State-of-the-Art in Driver's Drowsiness Detection: A Comprehensive Survey (2025) — https://www.researchgate.net/publication/390974266
- A Survey on State-of-the-Art Drowsiness Detection Techniques (behavioral/vehicular/physiological + ML comparison) — https://www.semanticscholar.org/paper/99dd774640fbd3ca1900259ab96d688e827946f1
- A Survey on Driver Drowsiness Detection (physiological, vehicular, behavioral) — https://www.researchgate.net/publication/259964148
- Background + commercial systems (Ford, Honda, etc.) — https://en.wikipedia.org/wiki/Driver_drowsiness_detection

### 4.2 Posture-based detection (core topic)
- Assessment of Driver's Drowsiness Based on Fractal Dimensional Analysis of Sitting and Back Pressure — https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.02362/full
- Analysis of Physiological Parameters and Driver Posture for Prevention of Road Accidents: A Review (2025) — https://www.mdpi.com/1424-8220/25/19/6238
- Machine Learning based Drowsiness Detection in Classrooms (sitting-posture index for weak drowsiness) — https://www.researchgate.net/publication/365252155
- Fatigue Detection via pose estimation (upper-body shoulder/neck joints vs. ideal posture) — https://arxiv.org/pdf/1911.10629
- Towards a New System for Drowsiness Detection Based on Eye Blinking and Head Posture Estimation — https://arxiv.org/pdf/1806.00360
- Head Pose Estimation and Micro-Expression Analysis for Driver Drowsiness Monitoring — https://www.researchgate.net/publication/400907240

### 4.3 Sitting-posture monitoring (background / pose tooling)
- Intelligent Systems for Sitting Posture Monitoring and Anomaly Detection: an overview — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10880321/
- Scene Recognition & Semantic Analysis for Unhealthy Sitting Posture Detection (Kinect + Faster R-CNN) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6163234/
- Automatic Detect Incorrect Lifting Posture with the Pose Estimation Model (pose + BiLSTM) — https://www.mdpi.com/2075-1729/15/3/358
- Industrial Ergonomics Risk Analysis Based on 3D Human Pose Estimation — https://www.mdpi.com/2079-9292/11/20/3403
- Automatic Real-Time Occupational Posture Evaluation (OpenPose) — https://www.nature.com/articles/s41598-022-05812-9

### 4.4 Eye / facial cues (EAR, PERCLOS, blink)
- Real-Time Drowsiness Detection Using Eye Aspect Ratio and Facial Landmark Detection — https://arxiv.org/pdf/2408.05836
- Eye Aspect Ratio for Real-Time Drowsiness Detection to Improve Driver Safety — https://www.mdpi.com/2079-9292/11/19/3183
- Real-Time Drivers' Drowsiness Detection and Analysis through Deep Learning (2025) — https://arxiv.org/pdf/2511.12438
- Long-term Multi-granularity Deep Framework for Driver Drowsiness Detection — https://arxiv.org/pdf/1801.02325
- Drowsiness Detection Based on Driver Temporal Behavior (CNN + LSTM) — https://arxiv.org/pdf/2104.00125
- A Real-Time Driver Drowsiness Detection System (AlexNet / VGG-FaceNet / FlowImageNet) — https://arxiv.org/pdf/2511.13618
- Deep CNN: Driver Drowsiness Detection Based on Eye State — https://www.researchgate.net/publication/338251837

### 4.5 Yawning / mouth cues (MAR)
- A Deep Learning Approach to Driver Fatigue Detection via Mouth-State & Yawning (ConNN, 99% on YawDD/NTHU/KouBM-DFD) — https://www.researchgate.net/publication/352... (search title; Academia: https://www.academia.edu/49040260)
- Vision-Based Driver Fatigue Detection Using Eye and Mouth Aspect Ratios with ML (2025) — https://www.researchgate.net/publication/391318509
- Automatic Fatigue Detection of Drivers through Yawning Analysis (degree-of-mouth-openness) — https://www.researchgate.net/publication/227240883
- AI-Powered Drowsiness and Yawning Detection for Proactive Driver Safety (2024, adds Facial Aspect Ratio) — https://www.researchgate.net/publication/387274245
- Real-Time Fatigue Detection Algorithms Using ML for Yawning and Eye State (2024 review) — https://www.researchgate.net/publication/386521644

### 4.6 Physiological & wearable (ECG / PPG / HRV / EMG / respiration)
- Using Wearable ECG/PPG Sensors — Recurrence Plots + CNN — https://www.mdpi.com/2079-9292/8/2/192
- Exploiting HRV for Driver Drowsiness Detection Using Wearable Sensors and ML (2025) — https://www.nature.com/articles/s41598-025-08582-2
- A New Method to Detect Driver Fatigue Based on EMG and ECG (portable non-contact, seat cushion) — https://www.researchgate.net/publication/320906647
- ECG-Based Driving Fatigue Detection Using HRV Analysis with Mutual Information (2023) — https://www.researchgate.net/publication/374397326
- Multi-Level Classification of Drowsiness by Simultaneous ECG + Respiration (CNN-LSTM) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9518416/

### 4.7 EEG-based deep learning
- A Deep Learning CNN Model for Driver Fatigue Detection Using a Single EEG Channel — https://www.academia.edu/49207293
- Drowsiness Detection Using EEG Signals and Machine Learning Algorithms (CNN, 94.75%) — https://www.researchgate.net/publication/360392717
- A CNN-Based Deep Learning Framework for Driver's Drowsiness (direct PDF) — https://thesai.org/Downloads/Volume15No3/Paper_17-A_CNN_based_Deep_Learning_Framework.pdf
- Drowsiness Detection of EEG Signals Using Image-Based CNN — https://www.researchgate.net/publication/376408281

### 4.8 Vehicle-based (steering, lane, braking)
- Real-Time Detection of Driver Fatigue Based on CNN-LSTM (IET, open) — https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12373
- A Lightweight Driver Drowsiness Detection System Using 3D-CNN with LSTM — https://www.researchgate.net/publication/366760642
- Drowsiness Monitoring by Steering and Lane Data Based Features — https://www.researchgate.net/publication/228951699
- Driver Drowsiness Detection Using Evolutionary ML (direct PDF) — https://www.bio-conferences.org/articles/bioconf/pdf/2024/16/bioconf_iscku2024_00007.pdf

### 4.9 Embedded / Raspberry Pi (real-time, low-cost)
- A Portable Fuzzy Driver Drowsiness Estimation System (PMC) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7435375/
- Driver Drowsiness Detection System Using Raspberry Pi (direct PDF) — https://www.ijsdr.org/papers/IJSDR2503258.pdf
- Driver Drowsiness Detection Using Raspberry Pi (direct PDF) — https://ijrcs.org/wp-content/uploads/IJRCS202005031.pdf
- Raspberry Pi-Based Driver Drowsiness Detection (87.8%) — https://www.researchgate.net/publication/379676181
- Drowsiness Detection Using Raspberry-Pi Based on Image Processing (Haar + Kalman + SVM) — https://www.researchgate.net/publication/348562766

### 4.10 Transformers / YOLO / newest deep learning
- Vision Transformers and YoloV5 based Driver Drowsiness Detection Framework — https://arxiv.org/pdf/2209.01401
- Applying Spatiotemporal Attention to Identify Distracted & Drowsy Driving with Vision Transformers — https://arxiv.org/pdf/2207.12148
- STFTransNet: Transformer-Based Spatial-Temporal Fusion Network (2025) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12473511/
- Integrating Lightweight YOLOv5s and Facial 3D Keypoints (Swin Transformer, 2024) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11784885/

### 4.11 Lightweight / IoT / cloud / mobile deployment
- IoT-Based Smart Alert System for Drowsy Driver Detection (InceptionV3 / VGG16 / MobileNetV2) — https://dl.acm.org/doi/abs/10.1155/2021/6627217
- IoT-Based Mobile Driver Drowsiness Detection Using Deep Learning (TinyML, SqueezeNet/MobileNet + quantization) — https://www.researchgate.net/publication/358912284
- The Driver Drowsiness Detection and Alert System Using IoT (ESP32-CAM + cloud, 92%, rest-stop suggestions) — https://www.researchgate.net/publication/390638540
- Design and Implementation of an IoT Drowsiness Detection System for Drivers — https://www.researchgate.net/publication/373343838

### 4.12 Datasets & benchmark studies (cross-dataset)
- Real-Time Driver Drowsiness Detection Using Facial Analysis & ML (benchmarks NTHU/YawDD/UTA-RLDD across KNN/SVM/CNN/YOLO) — https://www.mdpi.com/1424-8220/25/3/812
- Detection of Drowsiness Among Drivers Using a Novel Deep CNN Model — https://www.mdpi.com/1424-8220/23/21/8741
- Drivers Drowsiness Detection Using Condition-Adaptive Representation Learning Framework — https://arxiv.org/pdf/1910.09722
- Detecting Driver Drowsiness as an Anomaly Using LSTM Autoencoders — https://arxiv.org/pdf/2209.05269

---

## 5. Suggested implementation stack (for reference)

- **Language:** Python.
- **CV / landmarks:** OpenCV, Dlib (68-point shape predictor), or MediaPipe
  (Face Mesh + Pose), `imutils`.
- **Detection models:** YOLOv5/v8 (ROI/face), CNN (eye/mouth state), CNN-LSTM or
  3D-CNN (temporal), Vision Transformer (SOTA accuracy).
- **Classic ML baseline:** scikit-learn (SVM/KNN/RandomForest) on EAR/MAR/PERCLOS.
- **Posture:** MediaPipe Pose / OpenPose / BlazePose for upper-body joints; or
  seat-pressure sensor array if hardware-based.
- **Embedded:** Raspberry Pi + Pi camera; or ESP32-CAM with cloud offload; TinyML
  + quantization (TFLite) for on-device.
- **Alerting:** local alarm (pygame/buzzer), GSM/GPS, or cloud push + mobile app.

---

## 6. Notes / caveats

- Vision-only methods are non-intrusive and cheap but sensitive to lighting,
  occlusion (sunglasses, masks), and head pose — combining cues (eye + mouth +
  head/posture) improves robustness.
- Physiological methods (EEG especially) are accurate but intrusive; single-channel
  EEG and PPG-via-smartband are the practical compromises.
- Models trained only on NTHU-DDD may not generalize — validate across datasets
  and/or a custom set under varied lighting.
- This file is a research/engineering reference, not medical or safety-certified
  guidance.
