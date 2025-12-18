import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from colorama import Fore, Style, init

init(autoreset=True)

# ============================
# MODEL PATHS
# ============================
MODEL_PATH1 = "models/livestockvsunknown.tflite"
MODEL_PATH2 = "models/cattle_buffalo_effb3.tflite"

# Cattle
MODEL_PATH3 = "models/efficientnetb3_cattle_fp32.tflite"
LABELS_PATH_CATTLE = "models/labels.json"

# Buffalo
MODEL_PATH4 = "models/buffalo_fp32.tflite"
LABELS_PATH_BUFFALO = "models/buffalo_labels.txt"

TEST_DIR    = "test-images"
IMG_SIZE    = 300

# Thresholds
THRESHOLD_1 = 0.6   # livestock vs unknown
THRESHOLD_2 = 0.5   # cattle vs buffalo


# ============================
# LOAD TFLITE MODEL
# ============================
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()


# ============================
# IMAGE PRE-PROCESSING
# ============================
def preprocess_image(image_path, img_size=IMG_SIZE):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return img


# ============================
# PREDICT
# ============================
def predict_single(interpreter, input_details, output_details, tensor):
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]


# ============================
# MAIN CLASSIFICATION LOGIC
# ============================
def classify_image(
        image_name, img_tensor,
        # stage-1
        interp1, in1, out1,
        # stage-2
        interp2, in2, out2,
        # cattle
        interp3, in3, out3, cattle_labels,
        # buffalo
        interp4, in4, out4, buffalo_labels
    ):

    # -------- Stage-1 --------
    prob1 = float(predict_single(interp1, in1, out1, img_tensor)[0])
    if prob1 < THRESHOLD_1:
        print(
            f"{Fore.YELLOW}{image_name:35s}"
            f"{Fore.BLUE}UNKNOWN"
            f"{Style.RESET_ALL} (LivestockProb={prob1:.4f})"
        )
        return


    # -------- Stage-2 --------
    prob2 = float(predict_single(interp2, in2, out2, img_tensor)[0])
    is_cattle = prob2 > THRESHOLD_2

    if not is_cattle:
        species = "BUFFALO"
        color = Fore.GREEN
    else:
        species = "CATTLE"
        color = Fore.RED


    # -------- Breed Classification --------
    if species == "CATTLE":
        breed_probs = predict_single(interp3, in3, out3, img_tensor)
        breed_idx = int(np.argmax(breed_probs))
        breed_conf = float(breed_probs[breed_idx])
        breed_name = cattle_labels[str(breed_idx)]

        print(
            f"{Fore.YELLOW}{image_name:35s}"
            f"{Fore.MAGENTA}LIVESTOCK → {color}{species}"
            f"{Fore.CYAN} → {breed_name.upper()}{Style.RESET_ALL}"
            f" (L={prob1:.4f}, CattleProb={prob2:.4f}, BreedConf={breed_conf:.4f})"
        )
        return

    # ====== BUFFALO BREED ======
    breed_probs = predict_single(interp4, in4, out4, img_tensor)
    breed_idx = int(np.argmax(breed_probs))
    breed_conf = float(breed_probs[breed_idx])
    breed_name = buffalo_labels[breed_idx]

    print(
        f"{Fore.YELLOW}{image_name:35s}"
        f"{Fore.MAGENTA}LIVESTOCK → {color}{species}"
        f"{Fore.CYAN} → {breed_name.upper()}{Style.RESET_ALL}"
        f" (L={prob1:.4f}, CattleProb={prob2:.4f}, BreedConf={breed_conf:.4f})"
    )


# ============================
# RUN TEST
# ============================
def run_test():
    # Load models
    interp1, in1, out1 = load_tflite_model(MODEL_PATH1)
    interp2, in2, out2 = load_tflite_model(MODEL_PATH2)

    # Cattle model
    interp3, in3, out3 = load_tflite_model(MODEL_PATH3)

    # Buffalo model
    interp4, in4, out4 = load_tflite_model(MODEL_PATH4)

    # Load labels JSON (cattle)
    with open(LABELS_PATH_CATTLE, "r") as f:
        cattle_labels = json.load(f)

    # Load buffalo labels TXT
    with open(LABELS_PATH_BUFFALO, "r") as f:
        buffalo_labels = [line.strip() for line in f.readlines()]

    files = sorted(os.listdir(TEST_DIR))
    print(f"\n{Fore.CYAN}Found {len(files)} test images.\n")

    for name in files:
        path = os.path.join(TEST_DIR, name)
        img_tensor = preprocess_image(path)

        classify_image(
            name, img_tensor,
            interp1, in1, out1,
            interp2, in2, out2,
            interp3, in3, out3, cattle_labels,
            interp4, in4, out4, buffalo_labels
        )


if __name__ == "__main__":
    run_test()
