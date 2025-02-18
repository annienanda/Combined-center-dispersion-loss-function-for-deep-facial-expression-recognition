# Combined-center-dispersion-loss-function-for-deep-facial-expression-recognition

📌  Overview
This project implements our paper Combined center dispersion loss function for deep facial expression recognition. 
Paper link : https://www.sciencedirect.com/science/article/pii/S0167865520304074.
We implement Snapshot Ensemble & Incremental LR Scheduling with Combined Center Dispersion Loss (CCDL) for Facial Expression Recognition (FER). It uses VGGFace as the base model and fine-tunes it on the FERPlus dataset.

📌 The model:
Extracts facial features using VGGFace.
Uses CCDL to enhance feature representation by reducing intra-class variations and inter-class similarities.
Employs Snapshot Ensemble & Incremental LR Scheduling to train five different models with incremental learning rates, capturing diverse feature representations for better generalization.

📌 Project Structure
├── README.md                  # Project documentation (this file)
├── dataset.py                 # Loads and processes the FERPlus dataset
├── snapshot.py                # Implements Snapshot Ensemble callback
├── train.py                   # Trains 5 snapshot models and saves them
├── test.py                    # Loads 5 models, averages predictions, and evaluates performance

📌 Install Dependencies
pip install tensorflow keras numpy opencv-python

📌 Dataset Preparation
The model uses the FERPlus dataset for training and testing.
You can download the dataset from this link : https://github.com/microsoft/FERPlus
Make sure your dataset is structured as follows:

FERPLUS/
│── TRAIN/
│   ├── Neutral/
│   ├── Happiness/
│   ├── Surprise/
│   ├── Sadness/
│   ├── Anger/
│   ├── Disgust/
│   ├── Fear/
│
│── TEST/
│   ├── Neutral/
│   ├── Happiness/
│   ├── Surprise/
│   ├── Sadness/
│   ├── Anger/
│   ├── Disgust/
│   ├── Fear/

I have removed the class Contempt, making our problem a 7-class classification model.

🚀 Model Training

Run the following command to train 5 different models with Snapshot Ensemble Learning & Incremental LR:

python train.py

🔹 Training Details
Base Model: VGGFace
Loss Function: CCDL (Categorical Cross-Entropy + Center Loss + Equivalence Loss)
Optimizer: SGD (Stochastic Gradient Descent)
Snapshot Learning Rates: [0.013, 0.014, 0.015, 0.016, 0.017]
Epochs: 50 (5 snapshots trained for 10 epochs each)
Batch Size: 32

🔹 Output
After training, 5 snapshot models will be saved:
snapshot_model_1.h5
snapshot_model_2.h5
snapshot_model_3.h5
snapshot_model_4.h5
snapshot_model_5.h5

🚀 Testing & Inference

To evaluate the trained ensemble:

python test.py

🔹 Testing Process
Loads the 5 snapshot models.
Makes predictions on the test set.
Averages the predictions from all models.
Compute final accuracy.

🔹 Output Example

Loaded Snapshot Model 1.
Loaded Snapshot Model 2.
Loaded Snapshot Model 3.
Loaded Snapshot Model 4.
Loaded Snapshot Model 5.
Final Ensemble Test Accuracy: 81.83 %


📖 Explanation of Key Components

1️⃣ Snapshot Ensemble Learning & Incremental LR Scheduling
Instead of training multiple separate models, Snapshot Ensembles train a single model but save it at different points in its optimization trajectory using incrmental learning rates. This creates diverse models, which improves generalization.

2️⃣ CCDL (Combined Center Dispersion Loss)
To improve feature representation, we use CCDL, which includes:
Cross-Entropy Loss (for classification)
Center Loss (to reduce intra-class variance)
Dispersion Loss (to enhance discriminative features)
This combination helps the model learn better feature embeddings for emotion classification.

3️⃣ Why Use Snapshot Ensemble for Facial Expression Recognition?
FER datasets are small, and ensemble learning improves generalization.
Snapshot Ensemble reduces computational cost compared to training multiple separate models.
Different snapshots capture diverse features, leading to higher accuracy.

📊 Results
Model	Accuracy
Snapshot Ensemble (5 Models)	81.83 %


📩 Contact & Support

If you have any questions, feel free to reach out! 🚀

🔗 GitHub: annienanda/Combined-center-dispersion-loss-function-for-deep-facial-expression-recognition
📧 Email: an7081@gmail.com
