📊 Sentiment Analysis of Drug Reviews: A Comparative Study of ML and DL Models
🧠 Overview

This project is a graduate research thesis that compares traditional machine learning (ML) and deep learning (DL) models in analyzing drug reviews to assess drug effectiveness through sentiment classification. The core objective was to evaluate how well models like Random Forest, SVM, and XGBoost perform in comparison to LSTM and BERT on real-world user-generated health data.

The dataset was sourced from the UCI Machine Learning Repository and includes tens of thousands of drug reviews, along with associated user ratings, drug names, and medical conditions.
🧪 Goal

The research explored three key questions:

    How do traditional ML models compare to each other in classifying sentiment from drug reviews?

    How do DL models compare in the same task?

    Do DL models like BERT outperform traditional models, or vice versa?

📁 Dataset

    Source: UCI Drug Review Dataset

    Features: review, drugName, condition, rating

    Target Variable:

        class_2ways: Binary sentiment classification (positive if rating ≥ 5)

    Class Imbalance: Resolved using RandomOverSampler

🔨 Techniques and Tools
💻 Traditional ML Models

    Preprocessing: Lemmatization + TF-IDF

    Models: RandomForestClassifier, SVM, XGBoost

    Evaluation: Accuracy, Precision, Recall, F1-score

    Hyperparameter Tuning: GridSearchCV

⚙️ Deep Learning Models
🔹 LSTM

    Built with: Keras, TensorFlow

    Layers: Embedding → LSTM → Dense

    Techniques: Bidirectional layers, Dropout, EarlyStopping

🔹 BERT

    Framework: Hugging Face Transformers, PyTorch

    Model: bert-base-uncased fine-tuned for binary classification

    Tokenization: BertTokenizer

    Training: Custom Dataset class and DataLoader, linear learning rate scheduler

📈 Results
Model	Accuracy	Precision (Positive)	Recall (Negative)	F1-Score
Random Forest	97%	0.97	0.99	High
SVM	95%	0.97	0.97	High
XGBoost	94%	0.97	0.96	High
LSTM	82%	0.85	0.71	Medium
BERT	86%	0.90	0.68	Medium
🧠 Key Insight

Despite BERT’s potential in understanding language nuances, Random Forest outperformed all models in this binary classification task. However, BERT remains promising for future work involving nuanced or domain-specific sentiment.
🚧 Challenges

    Compute Constraints: Managed BERT’s memory demands using batch scheduling and model checkpointing.

    Data Imbalance: Addressed with RandomOverSampler.

    Evaluation Depth: Cross-model comparison helped interpret not just what worked, but why.

📚 What I Learned

    Fine-tuning BERT with PyTorch and Hugging Face

    Comparing model architectures for NLP

    Model evaluation and interpretability

    Managing preprocessing pipelines for both traditional and deep learning models
