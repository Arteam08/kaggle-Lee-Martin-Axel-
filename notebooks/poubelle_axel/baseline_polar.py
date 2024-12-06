####################################################################################################
####################################################################################################
####################################### import libraries ###########################################
####################################################################################################
####################################################################################################

import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import re
import time
import gensim.downloader as api
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tqdm

import pandas as pd
import polars as pl

import emot 
import emoji

from autocorrect import Speller
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from scipy.stats import mode


# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')
# Load GloVe model with Gensim's API
# embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings
# embeddings_model = api.load("fasttext-wiki-news-subwords-300")
embeddings_model = api.load("word2vec-google-news-300")

####################################################################################################
####################################################################################################
####################################### global variables ###########################################
####################################################################################################
####################################################################################################

spell=Speller(lang="en", fast=True)
list_of_countries_trigrams = ['AFG', 'RSA', 'ALB', 'ALG', 'GER', 'AND', 'ENG', 'ANG', 'AIA', 'ATG', 'KSA', 'ARG', 'ARM', 'ARU', 'AUS', 'AUT', 'AZE', 'BAH', 'BHR', 'BAN', 'BRB', 'BEL', 'BLZ', 'BEN', 'BER', 'BHU', 'BLR', 'MYA', 'BOL', 'BIH', 'BOT', 'BRA', 'BRU', 'BUL', 'BFA', 'BDI', 'CAM', 'CMR', 'CAN', 'CPV', 'CHI', 'CHN', 'CYP', 'COL', 'COM', 'CGO', 'PRK', 'KOR', 'CRC', 'CIV', 'CRO', 'CUB', 'CUR', 'DEN', 'DJI', 'DMA', 'SCO', 'EGY', 'UAE', 'ECU', 'ERI', 'ESP', 'EST', 'ESW', 'USA', 'ETH', 'FIJ', 'FIN', 'FRA', 'GAB', 'GAM', 'GEO', 'GHA', 'GIB', 'GRE', 'GRN', 'GUA', 'GUM', 'GUI', 'EQG', 'GNB', 'GUY', 'HAI', 'HON', 'HKG', 'HUN', 'CAY', 'COK', 'FRO', 'SOL', 'TCA', 'VGB', 'VIR', 'IND', 'IDN', 'IRQ', 'IRN', 'IRL', 'NIR', 'ISL', 'ISR', 'ITA', 'JAM', 'JPN', 'JOR', 'KAZ', 'KEN', 'KGZ', 'KVX', 'KUW', 'LAO', 'LES', 'LVA', 'LBN', 'LBR', 'LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MKD', 'MAD', 'MAS', 'MWI', 'MDV', 'MLI', 'MLT', 'MAR', 'MRI', 'MTN', 'MEX', 'MDA', 'MNG', 'MNE', 'MSR', 'MOZ', 'NAM', 'NEP', 'NCA', 'NIG', 'NGA', 'NOR', 'NCL', 'NZL', 'OMA', 'UGA', 'UZB', 'PAK', 'PLE', 'PAN', 'PNG', 'PAR', 'NED', 'WAL', 'PER', 'PHI', 'POL', 'PUR', 'POR', 'QAT', 'COD', 'CTA', 'DOM', 'CZE', 'ROU', 'RUS', 'RWA', 'SKN', 'SMR', 'VIN', 'LCA', 'SLV', 'SAM', 'ASA', 'STP', 'SEN', 'SRB', 'SEY', 'SLE', 'SIN', 'SVK', 'SVN', 'SOM', 'SDN', 'SSD', 'SRI', 'SWE', 'SUI', 'SUR', 'SYR', 'TJK', 'TAH', 'TPE', 'TAN', 'CHA', 'THA', 'TLS', 'TOG', 'TGA', 'TRI', 'TUN', 'TKM', 'TUR', 'UKR', 'URU', 'VAN', 'VEN', 'VIE', 'YEM', 'ZAM', 'ZIM', 'BOE', 'GUF', 'GBR', 'GLP', 'NMI', 'KIR', 'MTQ', 'NIU', 'REU', 'SMN', 'SMA', 'TUV', 'ZAN', 'ALA', 'COR', 'GRL', 'GUE', 'IMA', 'FLK', 'MHL', 'JER', 'MYT', 'FSM', 'MCO', 'PLW', 'EUS', 'ESH', 'BLM', 'SPM', 'SHN', 'VAT', 'WLF']

path_to_data = "../../challenge_data/"
path_to_training_tweets = path_to_data + "train_tweets"
path_to_eval_tweets = path_to_data + "eval_tweets"

# Initialize Python porter stemmer
ps = PorterStemmer()

# Initialize Python wordnet lemmatizer
wnl = WordNetLemmatizer()

# Initialize emot 
emot_obj = emot.core.emot() 

# Define a global variable for tracking progress
n_it = 0

# Total number of tweets to preprocess
total_tweets = 500000

####################################################################################################
####################################################################################################
######################################## functions #################################################
####################################################################################################
####################################################################################################

# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

def detect_emojis(text):
    """ 
    Detect emojis in a text

    Parameters:
        text (str): The text to detect emojis in
    Returns:
        dict: A dictionary containing the emojis and their meanings
    Examples:
        >>> detect_emojis('Python is 👍, ')
        {'value': ['👍'], 'mean': ['thumbs up']}
    """
    emojis = [char for char in text if char in emoji.EMOJI_DATA]
    meanings = [emoji.EMOJI_DATA[char]['en'] for char in emojis]
    return {'value': emojis, 'mean': meanings}

def transform_emojis_to_text(text):
    """
    Transform emojis to text

    Parameters:
        text (str): The text to transform
    Returns:
        str: The text with emojis transformed to text
    Examples:
        >>> transform_emojis_to_text('Python is 👍, ')
        'Python is thumbs up, '
    """
    emojis = detect_emojis(text)
    text_modified = text
    for i in range(len(emojis['value'])):
        text_modified = text_modified.replace(emojis['value'][i], emojis['mean'][i].replace('_', ' '))
    return text_modified

def transform_emoticons_to_text(text):
    """ 
    Transform emoticons to text
    
    Parameters:
        text (str): The text to transform
    Returns:
        str: The text with emoticons transformed to text
    Examples:
        >>> transform_emoticons_to_text('Python is :)') 
        'Python is smile'
    """
    emoticons = emot_obj.emoticons(text)
    text_modified = text 
    for i in range(len(emoticons['value'])):
        text_modified = text_modified.replace(emoticons['value'][i], emoticons['mean'][i])
    return text_modified

def preprocess_text(text, 
                    with_URLs=True,
                    with_trigrams=True,
                    with_spelling_correction=False,
                    with_emojis=True,
                    with_emoticons=True,
                    with_lowercasing=True,
                    with_punctuation=True,
                    with_numbers=True,
                    with_stopwords=True, 
                    with_lemmatization=True, 
                    with_stemming=True):
    global n_it
    n_it += 1
    if n_it % (total_tweets // 100) == 0 or n_it == total_tweets:  # Update every 1% or on the last iteration
        print(f"Processed {n_it}/{total_tweets} tweets ({n_it / total_tweets * 100:.1f}%)")
    if with_URLs:
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
    if with_trigrams:
        # if a word in the text is a country trigram, we replace it by 'country'
        text = re.sub(r'\b(?:' + '|'.join(list_of_countries_trigrams) + r')\b', 'country', text)
    if with_spelling_correction:
        # Correct spelling
        text = spell(text)
    if with_emojis:
        # Transform emojis to text
        text = transform_emojis_to_text(text)
    if with_emoticons:
        # Transform emoticons to text
        text = transform_emoticons_to_text(text)
    if with_lowercasing:
        # Lowercasing
        text = text.lower()
    if with_punctuation:
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
    if with_numbers:
        # Remove numbers
        text = re.sub(r'\d+', '', text)
    # Tokenization
    words = text.split()
    if with_stopwords:
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    if with_lemmatization:
        # Lemmatization
        words = [wnl.lemmatize(word) for word in words]
    if with_stemming:
        # Stemming
        words = [ps.stem(word) for word in words]
    return ' '.join(words)


print("\n" + 'Preprocessing text...' + '\n')
print("\n" + 'Opening the training files...' + '\n')

# Read all training files and concatenate them into one dataframe
li = []
for filename in os.listdir(path_to_training_tweets):
    df = pd.read_csv(path_to_training_tweets + "/" + filename)
    li.append(df)
df = pd.concat(li, ignore_index=True)

# KEEP_ONLY = 500000
# print(f'Keeping only the first {KEEP_ONLY} rows for faster processing...')
# df = df.head(KEEP_ONLY)

print('\n' + 'Applying preprocessing to each tweet...' + '\n')

# Apply preprocessing with progress bar
total_tweets = df.shape[0]
n_it = 0
df['Tweet'] = df['Tweet'].apply(preprocess_text)

print('\n' + 'Embedding the tweets...' + '\n')

# Apply preprocessing to each tweet and obtain vectors
vector_size = 300  # Adjust based on the chosen GloVe model
tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])
tweet_df = pd.DataFrame(tweet_vectors)

print('\n' + 'Grouping the tweets into their corresponding periods...' + '\n')

# Attach the vectors into the original dataframe
period_features = pd.concat([df, tweet_df], axis=1)
# Drop the columns that are not useful anymore
period_features = period_features.drop(columns=['Timestamp', 'Tweet'])

######## REMOVE THE PART WHERE WE GROUP THE TWEETS INTO THEIR CORRESPONDING PERIODS
# # Group the tweets into their corresponding periods. This way we generate an average embedding vector for each period
# period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

# We drop the non-numerical features and keep the embeddings values for each period
X = period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
# We extract the labels of our training samples
y = period_features['EventType'].values

print(f'X shape: {X.shape}')

###### Evaluating on a test set:

print('\n' + 'Evaluation on a test set...' + '\n')

# We split our data into a training and test set that we can use to train our classifier without fine-tuning into the
# validation set and without submitting too many times into Kaggle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training Dummy Classifier...")
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_clf.fit(X_train, y_train)
print("Dummy Classifier training complete.\n")

# Logistic Regression
print("Training Logistic Regression...")
logreg = LogisticRegression(
    random_state=42, max_iter=1000, penalty='l2', C=1.0, solver='lbfgs'
)
logreg.fit(X_train, y_train)
print("Logistic Regression training complete.\n")

# Decision Tree
print("Training Decision Tree...")
dectree = DecisionTreeClassifier(
    random_state=42, max_depth=10, min_samples_leaf=5, min_samples_split=10
)
dectree.fit(X_train, y_train)
print("Decision Tree training complete.\n")

# Random Forest
print("Training Random Forest...")
randfor = RandomForestClassifier(
    random_state=42, n_estimators=100, max_depth=10, max_features='sqrt', n_jobs=-1
)
randfor.fit(X_train, y_train)
print("Random Forest training complete.\n")

# # Gradient Boosting
# print("Training Gradient Boosting...")
# gradboost = GradientBoostingClassifier(
#     random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=4
# )
# gradboost.fit(X_train, y_train)
# print("Gradient Boosting training complete.\n")

# XGBoost
print("Training XGBoost...")
xgboost = XGBClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    tree_method='gpu_hist',  # Use GPU if available
    predictor='gpu_predictor',
    n_jobs=-1,
    eval_metric='mlogloss',
)
xgboost.fit(X_train, y_train)
print("XGBoost training complete.\n")

# # Support Vector Machine
# print("Training SVM...")
# svm = SVC(
#     random_state=42, C=1.0, kernel='rbf', gamma='scale', probability=True
# )
# svm.fit(X_train, y_train)
# print("SVM training complete.\n")

dummy_preds = dummy_clf.predict(X_test)
logreg_preds = logreg.predict(X_test)
dectree_preds = dectree.predict(X_test)
randfor_preds = randfor.predict(X_test)
# gradboost_preds = gradboost.predict(X_test)
xgboost_preds = xgboost.predict(X_test)
# svm_preds = svm.predict(X_test)

print("Dummy Classifier Accuracy: ", accuracy_score(y_test, dummy_preds))
print("Logistic Regression Accuracy: ", accuracy_score(y_test, logreg_preds))
print("Decision Tree Accuracy: ", accuracy_score(y_test, dectree_preds))
print("Random Forest Accuracy: ", accuracy_score(y_test, randfor_preds))
# print("Gradient Boosting Accuracy: ", accuracy_score(y_test, gradboost_preds))
print("XGBoost Accuracy: ", accuracy_score(y_test, xgboost_preds))
# print("SVM Accuracy: ", accuracy_score(y_test, svm_preds))

###### For Kaggle submission

print('\n' + 'Training the classifiers on the full dataset...' + '\n')

dummy_clf = DummyClassifier(strategy="most_frequent").fit(X, y)
logreg = LogisticRegression(
        random_state=42,
        max_iter=1000,
        penalty='l2',  # Use L2 regularization
        C=1.0,         # Regularization strength; smaller values specify stronger regularization
        solver='lbfgs' # Suitable solver for L2 penalty
    ).fit(X, y)
dectree = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,          # Limit the depth of the tree
        min_samples_leaf=5,    # Minimum number of samples required at a leaf node
        min_samples_split=10   # Minimum number of samples required to split an internal node
    ).fit(X, y)
randfor = RandomForestClassifier(
        random_state=42,
        n_estimators=200,       # Increase number of trees
        max_depth=15,           # Limit the depth of the trees
        max_features='sqrt',    # Number of features to consider when looking for the best split
        min_samples_leaf=4,     # Minimum number of samples required at a leaf node
        min_samples_split=10    # Minimum number of samples required to split an internal node
    ).fit(X, y)
# gradboost = GradientBoostingClassifier(
#         random_state=42,
#         n_estimators=300,       # Increase number of boosting stages
#         learning_rate=0.05,     # Shrinks the contribution of each tree
#         max_depth=5,            # Limit the depth of the trees
#         min_samples_leaf=4,     # Minimum number of samples required at a leaf node
#         min_samples_split=10    # Minimum number of samples required to split an internal node
#     ).fit(X, y)
xgboost = XGBClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        reg_lambda=1.0,        # L2 regularization term on weights
        subsample=0.8,         # Subsample ratio of the training instances
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        eval_metric='mlogloss'
    ).fit(X, y)
# svm = SVC(
#         random_state=42,
#         C=1.0,            # Regularization parameter
#         kernel='rbf',     # Radial basis function kernel
#         gamma='scale',    # Kernel coefficient
#         probability=True
#     ).fit(X, y)


dummy_predictions = []
logreg_predictions = []
dectree_predictions = []
randfor_predictions = []
# gradboost_predictions = []
xgboost_predictions = []
# svm_predictions = []

def majority_vote(group):
    """Compute the majority vote for a group of predictions."""
    if len(group) == 0:  # Handle empty groups
        return None
    
    # Use scipy's mode with keepdims=False to handle single-value or all-unique cases
    try:
        group_mode = mode(group, keepdims=False)
        return group_mode.mode
    except Exception as e:
        # If mode fails (e.g., all values are unique), return the most frequent value
        return group.value_counts().index[0]

for fname in os.listdir(path_to_eval_tweets):
    print('\n' + 'Opening the evaluation files...' + fname + '\n')

    val_df = pd.read_csv(path_to_eval_tweets + "/" + fname)
    # progress bar
    n_it = 0
    total_tweets = val_df.shape[0]
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)

    # print("\nDebug: After preprocessing tweets")
    # print("Shape of val_df:", val_df.shape)
    # print("Columns in val_df:", val_df.columns)

    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    # print("\nDebug: After embedding tweets")
    # print("Shape of tweet_df (embeddings):", tweet_df.shape)

    period_features = pd.concat([val_df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    # print("Shape of period_features:", period_features.shape)
    # print("Columns in period_features:", period_features.columns)
    # Extract the embedding features (last 300 columns)
    embedding_features = period_features.iloc[:, -300:]
    # print("Debug: Extracted embedding features shape:", embedding_features.shape)


    # Add prediction columns for each model
    period_features['DummyEventType'] = dummy_clf.predict(embedding_features)
    period_features['LogisticRegressionEventType'] = logreg.predict(embedding_features)
    period_features['DecisionTreeEventType'] = dectree.predict(embedding_features)
    period_features['RandomForestEventType'] = randfor.predict(embedding_features)
    # period_features['GradientBoostingEventType'] = gradboost.predict(embedding_features)
    period_features['XGBoostEventType'] = xgboost.predict(embedding_features)
    # period_features['SVMEventType'] = svm.predict(embedding_features)


    # Group by 'MatchID', 'PeriodID', 'ID' and apply majority vote
    dummy_majority = period_features.groupby(['MatchID', 'PeriodID', 'ID'])['DummyEventType'].apply(majority_vote).reset_index()
    logreg_majority = period_features.groupby(['MatchID', 'PeriodID', 'ID'])['LogisticRegressionEventType'].apply(majority_vote).reset_index()
    dectree_majority = period_features.groupby(['MatchID', 'PeriodID', 'ID'])['DecisionTreeEventType'].apply(majority_vote).reset_index()
    randfor_majority = period_features.groupby(['MatchID', 'PeriodID', 'ID'])['RandomForestEventType'].apply(majority_vote).reset_index()
    # gradboost_majority = period_features.groupby(['MatchID', 'PeriodID', 'ID'])['GradientBoostingEventType'].apply(majority_vote).reset_index()
    xgboost_majority = period_features.groupby(['MatchID', 'PeriodID', 'ID'])['XGBoostEventType'].apply(majority_vote).reset_index()
    # svm_majority = period_features.groupby(['MatchID', 'PeriodID', 'ID'])['SVMEventType'].apply(majority_vote).reset_index()

    # Append majority-voted predictions to corresponding lists
    dummy_predictions.append(dummy_majority)
    logreg_predictions.append(logreg_majority)
    dectree_predictions.append(dectree_majority)
    randfor_predictions.append(randfor_majority)
    # gradboost_predictions.append(gradboost_majority)
    xgboost_predictions.append(xgboost_majority)
    # svm_predictions.append(svm_majority)

# Concatenate all predictions for each model
pd.concat(dummy_predictions).to_csv('dummy_majority_predictions.csv', index=False)
pd.concat(logreg_predictions).to_csv('logistic_majority_predictions.csv', index=False)
pd.concat(dectree_predictions).to_csv('decision_tree_majority_predictions.csv', index=False)
pd.concat(randfor_predictions).to_csv('random_forest_majority_predictions.csv', index=False)
# pd.concat(gradboost_predictions).to_csv('gradient_boosting_majority_predictions.csv', index=False)
pd.concat(xgboost_predictions).to_csv('xgboost_majority_predictions.csv', index=False)
# pd.concat(svm_predictions).to_csv('svm_majority_predictions.csv', index=False)

print("Majority voting and prediction saving complete.")
