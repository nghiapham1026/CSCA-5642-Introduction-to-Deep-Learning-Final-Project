# **Sentiment Analysis of IMDB Movie Reviews Using Deep Learning**

## **Project Overview**
This project focuses on classifying the sentiment of IMDB movie reviews as either positive or negative using various Deep Learning techniques. The dataset contains 50,000 labeled reviews and serves as a benchmark for natural language processing (NLP) tasks. The project explores and compares the performance of three models:
1. A Baseline Feedforward Neural Network.
2. A Long Short-Term Memory (LSTM) network.
3. A fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model.

Key components include:
- Data cleaning and preprocessing.
- Exploratory Data Analysis (EDA).
- Model training, hyperparameter tuning, and evaluation.
- Comparative analysis and visualizations.

---

## **Dataset**
- **Source**: [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Description**:
  - 50,000 movie reviews labeled as either positive or negative.
  - Balanced dataset: 25,000 positive and 25,000 negative reviews.
  - Text data with varying review lengths and sentiment labels.

To download the dataset, follow the link above or run the preprocessing scripts to load and clean the data.

---

## **How to Use**
### 1. **Setup the Environment**
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

### 2. **Data Preprocessing**
Run the preprocessing script to clean and tokenize the dataset:
```bash
python scripts/data_preprocessing.py
```

This will save the preprocessed data to the `data/processed/` directory.

---

### 3. **Train Models**
Train each model using the provided scripts:
- **Baseline Model**:
  ```bash
  python scripts/train_baseline.py
  ```
- **LSTM Model**:
  ```bash
  python scripts/train_lstm.py
  ```
- **BERT Model**:
  ```bash
  python scripts/train_bert.py
  ```

Trained models will be saved in the `models/` directory.

---

### 4. **Evaluate Models**
Run the evaluation script to compare models:
```bash
python scripts/evaluate_models.py
```

This script will print classification metrics (e.g., accuracy, F1-score) and display visualizations such as confusion matrices and ROC curves.

---

## **Results**
| Model          | Test Accuracy | Validation Accuracy | AUC Score |
|-----------------|---------------|---------------------|-----------|
| Baseline Model | 83.96%        | 83.60%             | 0.84      |
| LSTM Model     | 85.40%        | 83.66%             | 0.85      |
| BERT Model     | 90.86%        | 90.34%             | 0.91      |

The **BERT model** achieved the highest performance, demonstrating the value of pre-trained embeddings and contextual understanding of text.

---

## **Key Visualizations**
1. **Confusion Matrices**:
   - Baseline, LSTM, and BERT models.
2. **ROC Curves**:
   - Comparison of model performance across thresholds.
3. **Training Curves**:
   - Accuracy and loss trends over epochs for each model.

---

## **Future Work**
- Experiment with additional pre-trained models such as RoBERTa or DistilBERT.
- Explore multi-class sentiment classification (e.g., neutral, positive, negative).
- Incorporate metadata features (e.g., review length or user ratings) for multi-modal analysis.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.