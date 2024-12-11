# kaggle-Lee-Martin-Axel

*Authors : Axel Delaval, Martin Drieux, Lee Kadz*

# Project: Sub-Event Detection in Football Matches

This project is part of a machine learning challenge to build a binary classification model that predicts notable events (like goals or red cards) during football matches, based on tweet data from the 2010 and 2014 FIFA World Cups. The main objective is to classify each one-minute interval of a match, using tweets, as containing a notable event or not.

---

## Repository Structure

- **`preprocessing.ipynb`**: Jupyter Notebook for preprocessing the data, including data cleaning, normalization, and feature extraction. The processed datasets are saved as:
  - `df_train_withoutstemming.csv`: Training dataset after preprocessing.
  - `df_eval_withoutstemming.csv`: Evaluation dataset after preprocessing.
- **`LSTM.ipynb`**: Notebook implementing the LSTM embedding method and training/evaluating machine learning models using this embedding.
- **`TF-IDF.ipynb`**: Notebook implementing the TF-IDF embedding method and training/evaluating machine learning models using this embedding.
- **`BERTweet.ipynb`**: Notebook utilizing the BERTweet embedding method and training/evaluating machine learning models using this embedding.

---

## Usage

### 1. Preprocessing
Open and execute the `preprocessing.ipynb` notebook to prepare the dataset for embedding and model training. The processed datasets are saved as:
  - `df_train_withoutstemming.csv`: Training dataset after preprocessing.
  - `df_eval_withoutstemming.csv`: Evaluation dataset after preprocessing.

### 2. Embedding and Model Training
Use one of the embedding notebooks to both generate features and train/evaluate the models:
- **LSTM embedding and models**: Run `LSTM.ipynb`.
- **TF-IDF embedding and models**: Run `TF-IDF.ipynb`.
- **BERTweet embedding and models**: Run `BERTweet.ipynb`.

Each notebook includes the following machine learning models:
1. **Neural Networks (NN)**: Deep learning model with customizable architecture.
2. **Recurrent Neural Networks (RNN)**: Sequence-based model using LSTM layers.
3. **Support Vector Machines (SVM)**: Classical machine learning model for classification tasks.
4. **Logistic Regression (LG)**: Simple and interpretable linear model.
5. **Random Forest (RF)**: Ensemble learning method using decision trees.
6. **Extreme Gradient Boosting (XGB)**: Advanced gradient boosting framework.

---

## Results

Performance metrics (e.g., accuracy) for each embedding-model combination are calculated and displayed within the notebooks.


