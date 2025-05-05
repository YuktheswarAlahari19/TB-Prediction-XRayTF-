# TB Prediction from Chest X-Ray Images

## Project Overview

This repository demonstrates a TensorFlow-based pipeline for analyzing chest X-ray images to predict tuberculosis (TB). We sourced TB chest X-ray images from Kaggle and used TensorFlow to build an end-to-end workflow, from data ingestion and preprocessing (including Contrast Limited Adaptive Histogram Equalization, CLAHE) through model training and evaluation.

## Dataset

* **Source**: Kaggle TB Chest X-Ray Images Dataset
* 
**Classes:**

- Healthy
- Tuberculosis (TB)

## Key Components

1. **Data Exploration & Preprocessing**

   * Visualized class distribution and image samples in `exploration.ipynb`.
   * Applied standard resizing and normalization.
   * Enhanced image contrast using CLAHE to improve feature visibility before feeding images into the model.
2. **TensorFlow Pipeline**

   * Created a TensorFlow pipeline for efficient batch loading, shuffling, and augmentation.
   * CLAHE preprocessing integrated as a custom TensorFlow map function.
3. **Model Architecture**

   * Custom CNN defined in `exploration.ipynb`, designed for grayscale X-ray inputs.
   * Experimented with deeper layers and dropout for regularization.
4. **Training & Evaluation**

   * managing training loops, learning rate scheduling, and checkpointing.
   * Metrics tracked include accuracy, precision, recall, and F1-score.
   * Generated confusion matrices and learning curves .

## Installation & Usage

1. **Clone & Setup**:

   ```bash
   git clone https://github.com/YuktheswarAlahari19/TB-Prediction-XRayTF-.git
   cd TB-Prediction-XRayTF-
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Explore & Preprocess**:

   * Launch `exploration.ipynb` to review preprocessing steps and CLAHE implementation.
3. **Train**:

   ```bash
   python src/train.py --data_dir data/ --epochs 50 --batch_size 32 --learning_rate 1e-3
   ```
4. **Evaluate**:

   * Review printed metrics and saved plots in `models/`.

## Results & Next Steps

* Achieved validation accuracy of \~X%.
* Future work: explore advanced architectures (e.g., EfficientNet), additional augmentations, and hyperparameter tuning.

## Contributing

Contributions are welcome—feel free to open issues or pull requests for improvements.

