# HybridColourNet-Vehicle-Detection-Color-Classification

A fun project exploring vehicle detection and color classification using a hybrid CNN+KNN approach.

Project Overview
-

HybridColourNet is a computer vision system I developed to better understand how different machine learning techniques can be combined for real-world applications. The project focuses on detecting vehicles and accurately classifying their colours using a fusion of YOLO-based detection with CNN and KNN classification.

Core Components
-

**1. Vehicle Detection: LEAF-YOLO (pre-trained)**

LEAF-YOLO is a recent and efficient object detection algorithm specifically designed for challenges like detecting small objects in complex aerial imagery captured by drones.

Reasons for using LEAF-YOLO:
- Lightweight: Small model size for high efficiency.
- Proven Accuracy: Demonstrated superior performance on the VisDrone benchmark.
- Real-Time Capable: Designed for fast inference on edge devices.
- Modern Architecture: A recent (2025) advancement over older YOLO variants.

Although I did train the model separately, I ultimately chose to use the pretrained model provided by the authors due to its results being more robust, especially in crowded settings.

**2. Colour Classification Model**
   

**CNN:** This is a small, custom-built neural network trained end-to-end on the dataset.
A CNN can learn deep visual patterns such as shading, texture, lighting conditions, and localized colour cues.

However, the CNN struggles with:
-Grey vs. Black/White: These colors share overlapping brightness values, causing confusion.
-“Other” category: This class contains many unrelated colors, making it less consistent and harder to learn.
-Far-distance vehicles: Small crops mean fewer pixels → CNN loses detail and becomes less confident.

[insert results]

**KNN:** The K-Nearest Neighbors (KNN) model is used as a lightweight, fast, and interpretable baseline for vehicle colour classification.
Instead of learning features, it classifies each vehicle by comparing its mean Lab colour (a 3-value feature) with the closest samples in the training set.

However, it struggles with:
- "grey" as it overlaps with black and white in brightness, causing those colours to blend together in Lab space.
- "other" class because it contains many unrelated colours (yellow, green, brown, beige…), so its samples are widely scattered.

[insert results] 

**CNN+KNN Fusion:** The goal is to combine the strengths of the two very different colour classification models to reduce their weaknesses. 
CNN is highly accurate when the object is large, clear and well lit. 
KNN is more stable when vehicles are small, far waay or blurry, because it relies only on simple colour statistics.

**How it works:**

1. The CNN predicts a colour and provides a confidence score.
2. The KNN predicts a colour based on similarity in Lab colour space.
3. The fusion rule then selects the final colour using:
- High CNN confidence → trust CNN
- Low CNN confidence → trust KNN
- If both agree → accept the shared answer
4. Special handling for grey (trust KNN for distant low-detail cars, and trust CNN when distinguishing black/white up close)

[maybe insert results?]

**3. Dataset**

This project is based on the **VCOR – Vehicle Color Recognition Dataset** from Kaggle. I adapted the original dataset for a 6-class vehicle colour task ```black``` ```white``` ```grey``` ```blue``` ```red``` ```other```, as well as the following additional changes:
- class balancing: revening out the number of samplesper colour class
- simple data augmentaion: randomly selected 25% of the dataset and augmented with one of the options below:
  - Horizontal flip
  - Gaussian blur
  - Random corner crop (keeps ~85% of the original width/height, chosen from one of the four corners)

Installation and Setup
-

1. Create and activate conda environment.

```
conda create -n hybridcolournet python=3.9
conda activate hybridcolournet
```

3. Clone repo
   
```
to insert later
```

5. Clone LEAF-YOLO repo **inside** project.

Send them to the link to follow the guide

7. Install requirements

```
pip install -r requirements.txt
```

Dataset Requirements
-

In order to use your own dataset, it must follow the structure outlined below:

```
dataset/
├── train/
│   ├── black/
│   ├── white/
│   ├── grey/
│   ├── blue/
│   ├── red/
│   └── other/
└── test/
    ├── black/
    ├── white/
    ├── grey/
    ├── blue/
    ├── red/
    └── other/
```

Training/Testing CNN
-

1. Train CNN

```
python scripts/train_vehicle_color.py --train-dir "path/to/dataset/train"
```
3. Test CNN

```
python scripts/test_vehicle_color.py --train-dir "path/to/dataset/test"
```

Building/Testing KNN
- 

1. Build KNN dataset

```
python scripts/build_color_knn_dataset.py --train-root "path/to/dataset/train"
```

3. Test KNN:

```
python scripts/test_knn.py --test-root "path/to/dataset/test"
```

Inference
-
The script uses the following default configuration. See the table below for default values. 

| Argument | Default Value |
| :--- | :--- |
| `--source` | `"inference/images"` |
| `--weights` | `["cfg/LEAF-YOLO/leaf-sizes/weights/best.pt"]` |
| `--color-model` | `"results/cnn/vehicle_color_best.h5"` |
| `--knn-data` | `None` |
| `--color` | `"all"` |
| `--img-size` | `640` |
| `--conf-thres` | `0.25` |
| `--iou-thres` | `0.45` |
| `--device` | `""` |
| `--project` | `"results/cnn_knn/car_finder"` |
| `--name` | `"exp"` |

1. For an image (default):
   
```
python scripts/car_finder_cnn_knn.py --source "image.jpg"
```

2. For a video (default + colour specification eg.):

```
python scripts/car_finder_cnn_knn.py --source "video.mp4" --color "red"
```

3. For Custom Configuration:

```
python scripts/car_finder_cnn_knn.py \
    --source <path_to_input_image_video_or_folder> \
    --weights <path_to_yolo_weights_file.pt> \
    --color-model <path_to_cnn_model_file.h5> \
    --knn-data <path_to_knn_data_file.npz> \
    --color <target_color_or_'all'>
```

Or single command:

```
python scripts/car_finder_cnn_knn.py --source <path_to_input> --weights <path_to_weights.pt> --color-model <path_to_cnn.h5> --knn-data <path_to_knn.npz> --color <target_color>
```




   
