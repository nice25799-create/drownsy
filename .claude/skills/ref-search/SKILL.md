---
name: ref-search
description: Search the project's reference papers (extracted text in reference/_text/) for a topic like PERCLOS, MAR, head pose, or CNN, and summarize what each paper says with citations.
---

The user provides a topic or keyword as the argument (e.g. "PERCLOS", "yawning", "calibration", "random forest").

1. The reference papers live as PDFs in `D:\project\reference\` with pre-extracted plain text in `D:\project\reference\_text\*.txt`. Always search the `_text` folder, not the PDFs.
2. Use the Grep tool with `-i` and a few lines of context (`-C 3`) to find the topic across all txt files. Try synonyms too (e.g. PERCLOS / "percentage of eyelid closure"; MAR / "mouth aspect ratio"; head pose / nodding).
3. Read the surrounding sections of the strongest matches to understand the context — not just the matching line.
4. Summarize per paper: which paper says what about the topic, including concrete numbers (thresholds, accuracies, window sizes) when present. Cite the paper filename so the user can find it.
5. If the topic appears nowhere, say so and suggest the closest related terms that do appear.

Paper key (filename -> what it is):
- `Real-time_eye_blink_detection_using_general_camera.txt` — Lu 2023, EAR blink detection with general cameras
- `jimaging-09-00091-v3.txt` — Albadawi 2023, EAR+MAR+head pose -> RF/NN/SVM, 99% on NTHU-DDD
- `sensors-22-02069.txt` — Albadawi 2022, survey of drowsiness detection systems
- `State-of-the-Art_in_Drivers_Drowsiness_Detection_.txt` — survey
- `bioconf_iscku2024_00007.txt` — Jumhaa 2024, survey of ML/DL + evolutionary algorithms
- `Kazemi_One_Millisecond_Face_2014_CVPR_paper.txt` — the dlib 68-landmark regression trees algorithm
- `05 (1).txt` — Rupani et al., basic EAR + dlib system
- `2408.05836v1.txt` — basic EAR + facial landmark system
