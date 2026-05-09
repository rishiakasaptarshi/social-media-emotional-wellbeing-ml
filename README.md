# 📱 Social Media Usage and Emotional Well-Being

> A comprehensive machine learning analysis exploring the relationship between social media usage patterns and emotional well-being using traditional ML, deep learning, and transformer models.

---

## 📊 Project Overview

This project employs multiple machine learning and deep learning approaches to analyze how social media usage impacts emotional well-being. It compares three distinct modeling approaches:

| Model | Type | Purpose |
|-------|------|---------|
| **Random Forest Classifier** | Traditional ML | Baseline classification |
| **Deep Learning Neural Network** | Deep Learning | Advanced pattern recognition |
| **DistilBERT Transformer** | Transfer Learning | State-of-the-art NLP classification |

---

## 🎯 Key Features

- ✅ **Multi-Model Comparison**: Benchmarks traditional ML vs. deep learning vs. transformer models
- ✅ **Transfer Learning**: Leverages pre-trained Hugging Face DistilBERT model
- ✅ **Comprehensive Dataset**: Uses Kaggle dataset with train/validation/test splits
- ✅ **Performance Metrics**: Detailed evaluation with accuracy, precision, recall, F1-scores
- ✅ **Model Persistence**: Trained models saved for inference and deployment

---

## 📦 Dataset

**Source**: [Social Media Usage and Emotional Well-Being - Kaggle](https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being/)

**Dataset Files**:
```
train.csv    - Training dataset
val.csv      - Validation dataset
test.csv     - Test dataset
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone or download** the project:
```bash
cd "Social Media Usage and Emotional Well-Being"
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download the dataset**:
   - Visit: https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being/
   - Download the ZIP file
   - Extract and copy these files to the project directory:
     - `train.csv`
     - `val.csv`
     - `test.csv`

4. **Run the project**:
```bash
python main.py
```

---

## 📁 Project Structure

```
Social Media Usage and Emotional Well-Being/
├── main.py                          # Main entry point
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── train.csv                        # Training data
├── val.csv                          # Validation data
├── test.csv                         # Test data
├── models/
│   └── deep_learning_model.h5       # Saved neural network model
└── outputs/
    └── model_comparison.csv         # Performance metrics
```

---

## 🔧 Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest)
- **Deep Learning**: TensorFlow/Keras
- **Transfer Learning**: Hugging Face Transformers, DistilBERT
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: Scikit-learn metrics

---

## 📈 Expected Results

The project generates:
- Model performance comparisons
- Classification metrics (accuracy, precision, recall, F1-score)
- Saved trained models in `/models/`
- Results summary in `/outputs/model_comparison.csv`

---

## 📝 Requirements

See `requirements.txt` for complete dependencies.

---

## 💡 Notes

- Ensure CSV files are properly extracted and placed in the project root
- First run may take time as Hugging Face transformer models are downloaded
- GPU recommended for faster training (CUDA enabled TensorFlow)

---

## 📄 License

This project is part of an academic final project.
