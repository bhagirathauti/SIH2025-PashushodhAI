🐄🐃 Multi-Stage Livestock Classification Pipeline (TensorFlow Lite)

A production-style computer vision pipeline built using TensorFlow Lite, designed for real-time, edge-device inference.
The system uses four lightweight AI models arranged in a gated, multi-stage decision flow to efficiently classify livestock images.

Although trained on livestock data, the architecture and inference strategy directly apply to dashcam, video analytics, and telematics systems.

🚀 Project Overview

Instead of relying on a single large model, this project follows a hierarchical inference approach:

Filter invalid or irrelevant inputs

Identify livestock species

Predict fine-grained class (breed)

This design improves:

Accuracy

Latency

Interpretability

Edge-device performance

🧠 Pipeline Architecture (4 Models)
Input Image
   ↓
Model 1 → Livestock vs Unknown
   ↓
Model 2 → Cattle vs Buffalo
   ↓
Model 3 → Cattle Breed Classifier
        OR
Model 4 → Buffalo Breed Classifier
   ↓
Final Prediction + Confidence Scores


Each model solves one focused task, reducing false positives and improving reliability.

🧩 Models Included
Model 1: Livestock vs Unknown

Filters out non-livestock images

Prevents invalid predictions early

Output: Livestock / Unknown

Model 2: Cattle vs Buffalo

Determines livestock species

Routes image to the correct downstream classifier

Output: Cattle / Buffalo

Model 3: Cattle Breed Classifier

Predicts cattle breed

EfficientNet-based architecture

Output: Breed name + confidence

Model 4: Buffalo Breed Classifier

Predicts buffalo breed

Optimized for Indian buffalo breeds

Output: Breed name + confidence

📁 Repository Structure
📦 project-root
 ┣ 📂 model_notebooks
 ┃ ┣ buffalo-classifier.ipynb
 ┃ ┣ cattlevsbuffalo.ipynb
 ┃ ┣ cow-classifier.ipynb
 ┃ ┗ livestockvsunknown.ipynb
 ┣ 📂 models
 ┃ ┣ livestockvsunknown.tflite
 ┃ ┣ cattle_buffalo_effb3.tflite
 ┃ ┣ efficientnetb3_cattle_fp32.tflite
 ┃ ┣ buffalo_fp32.tflite
 ┃ ┣ labels.json
 ┃ ┗ buffalo_labels.txt
 ┣ 📂 test-images
 ┃ ┗ *.jpg / *.png
 ┣ 📄 main.py
 ┣ 📄 requirements.txt
 ┗ 📄 README.md

⚙️ Inference Logic (main.py)

Loads all four TensorFlow Lite models

Preprocesses images to 300×300 RGB

Normalizes pixel values to 0–1 range

Applies confidence-based gating at each stage

Prints structured predictions with confidence scores

Thresholds Used
THRESHOLD_1 = 0.6   # Livestock vs Unknown
THRESHOLD_2 = 0.5   # Cattle vs Buffalo

▶️ Running the Pipeline
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Add test images

Place images inside:

test-images/

3️⃣ Run inference
python main.py

📊 Sample Output
cow_01.jpg        LIVESTOCK → CATTLE → GIR
(L=0.91, CattleProb=0.84, BreedConf=0.88)

buff_07.jpg       LIVESTOCK → BUFFALO → MURRAH
(L=0.94, CattleProb=0.12, BreedConf=0.90)

random.jpg        UNKNOWN (LivestockProb=0.23)

🧠 Relevance to Real-Time Video & Telematics Systems

While this project focuses on livestock imagery, the core architecture is directly transferable to:

Dashcam video pipelines

Driver behavior analysis

Safety event detection

Real-time alert systems

Equivalent Dashcam Flow Example:

Frame → Valid Scene? → Object Type? → Risk Classification → Alert


Key similarities:

Multi-stage filtering

Lightweight edge inference

Threshold-based decision making

Modular model updates

⚡ Performance & Optimization

Framework: TensorFlow Lite

Designed for offline / edge deployment

Modular models enable:

Faster inference

Easier maintenance

Independent retraining

🔮 Future Enhancements

View-angle validation (front / side / back)

Cross-breed detection

Temporal smoothing for video streams

Full INT8 quantization

Streaming inference support

👨‍💻 Author

Bhagirath Auti
AI / ML & Full-Stack Developer
🏆 Smart India Hackathon 2025 Winner

GitHub: https://github.com/bhagirathauti

LinkedIn: https://www.linkedin.com/in/bhagirathauti/

This repository demonstrates real-world ML system design, not just model training.
The staged inference approach, TFLite deployment, and confidence-based routing mirror production computer vision pipelines used in dashcam and telematics systems.