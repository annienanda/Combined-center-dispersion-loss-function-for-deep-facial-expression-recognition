# Combined-center-dispersion-loss-function-for-deep-facial-expression-recognition

📌  Overview
This project implements our paper Combined center dispersion loss function for deep facial expression recognition.<br> 
Paper link : https://www.sciencedirect.com/science/article/pii/S0167865520304074.<br>
We implement Snapshot Ensemble & Incremental LR Scheduling with Combined Center Dispersion Loss (CCDL) for Facial Expression Recognition (FER). It uses VGGFace as the base model and fine-tunes it on the FERPlus dataset.<br>

📌 The model:<br>
Extracts facial features using VGGFace.<br>
Uses CCDL to enhance feature representation by reducing intra-class variations and inter-class similarities.<br>
Employs Snapshot Ensemble & Incremental LR Scheduling to train five different models with incremental learning rates, capturing diverse feature representations for better generalization.<br>

📌 Project Structure<br>
├── README.md                  # Project documentation (this file)<br>
├── dataset.py                 # Loads and processes the FERPlus dataset<br>
├── snapshot.py                # Implements Snapshot Ensemble callback<br>
├── train.py                   # Trains 5 snapshot models and saves them<br>
├── test.py                    # Loads 5 models, averages predictions, and evaluates performance<br>

📌 Install Dependencies<br>
pip install tensorflow keras numpy opencv-python<br>

📌 Dataset Preparation<br>
The model uses the FERPlus dataset for training and testing.<br>
You can download the dataset from this link : https://github.com/microsoft/FERPlus<br>
Make sure your dataset is structured as follows:<br>

FERPLUS/<br>
│── TRAIN/<br>
│   ├── Neutral/<br>
│   ├── Happiness/<br>
│   ├── Surprise/<br>
│   ├── Sadness/<br>
│   ├── Anger/<br>
│   ├── Disgust/<br>
│   ├── Fear/<br>
│<br>
│── TEST/<br>
│   ├── Neutral/<br>
│   ├── Happiness/<br>
│   ├── Surprise/<br>
│   ├── Sadness/<br>
│   ├── Anger/<br>
│   ├── Disgust/<br>
│   ├── Fear/<br>

I have removed the class Contempt, making our problem a 7-class classification model.<br>

🚀 Model Training<br>

Run the following command to train 5 different models with Snapshot Ensemble Learning & Incremental LR:<br>

python train.py<br>

🔹 Training Details<br>
Base Model: VGGFace<br>
Loss Function: CCDL (Categorical Cross-Entropy + Center Loss + Equivalence Loss)<br>
Optimizer: SGD (Stochastic Gradient Descent)<br>
Snapshot Learning Rates: [0.013, 0.014, 0.015, 0.016, 0.017]<br>
Epochs: 50 (5 snapshots trained for 10 epochs each)<br>
Batch Size: 32<br>

🔹 Output<br>
After training, 5 snapshot models will be saved:<br>
snapshot_model_1.h5<br>
snapshot_model_2.h5<br>
snapshot_model_3.h5<br>
snapshot_model_4.h5<br>
snapshot_model_5.h5<br>

🚀 Testing & Inference<br>

To evaluate the trained ensemble:<br>

python test.py<br>

🔹 Testing Process<br>
Loads the 5 snapshot models.<br>
Makes predictions on the test set.<br>
Averages the predictions from all models.<br>
Compute final accuracy.<br>

🔹 Output Example<br>

Loaded Snapshot Model 1.<br>
Loaded Snapshot Model 2.<br>
Loaded Snapshot Model 3.<br>
Loaded Snapshot Model 4.<br>
Loaded Snapshot Model 5.<br>
Final Ensemble Test Accuracy: 81.83 %<br>


📖 Explanation of Key Components<br>

1️⃣ Snapshot Ensemble Learning & Incremental LR Scheduling<br>
Instead of training multiple separate models, Snapshot Ensembles train a single model but save it at different points in its optimization trajectory using incrmental learning rates. This creates diverse models, which improves generalization.<br>

2️⃣ CCDL (Combined Center Dispersion Loss)<br>
To improve feature representation, we use CCDL, which includes:<br>
Cross-Entropy Loss (for classification)<br>
Center Loss (to reduce intra-class variance)<br>
Dispersion Loss (to enhance discriminative features)<br>
This combination helps the model learn better feature embeddings for emotion classification.<br>

3️⃣ Why Use Snapshot Ensemble for Facial Expression Recognition?<br>
FER datasets are small, and ensemble learning improves generalization.<br>
Snapshot Ensemble reduces computational cost compared to training multiple separate models.<br>
Different snapshots capture diverse features, leading to higher accuracy.<br>

📊 Results<br>
Model	Accuracy<br>
Snapshot Ensemble (5 Models):	81.83 %<br>


📩 Contact & Support<br>

If you have any questions, feel free to reach out! <br>

🔗 GitHub: annienanda/Combined-center-dispersion-loss-function-for-deep-facial-expression-recognition<br>
📧 Email: an7081@gmail.com<br>
