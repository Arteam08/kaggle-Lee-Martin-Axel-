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
embeddings_model = api.load("fasttext-wiki-news-subwords-300")

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

# Load data using Polars
print("\nOpening the training files...\n")
dfs = [pl.read_csv(f"{path_to_training_tweets}/{filename}") for filename in os.listdir(path_to_training_tweets)]
df = pl.concat(dfs)

KEEP_ONLY = 10000
df = df.head(KEEP_ONLY)

# Preprocess tweets
print("\nApplying preprocessing to each tweet...\n")
df = df.with_column(
    pl.col("Tweet").apply(preprocess_text).alias("Tweet")
)

# Embedding tweets
print("\nEmbedding the tweets...\n")
tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, 300) for tweet in df["Tweet"].to_list()])
tweet_df = pl.DataFrame(tweet_vectors)

# Combine embeddings with the main DataFrame
df = pl.concat([df, tweet_df], how="horizontal").drop(["Timestamp", "Tweet"])

# Prepare features and labels
X = df.drop(["EventType", "MatchID", "PeriodID", "ID"]).to_numpy()
y = df["EventType"].to_numpy()

print(f"X shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define and train classifiers
classifiers = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_leaf=5),
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=15),
    "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=300, learning_rate=0.05),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=300, learning_rate=0.05, use_label_encoder=False),
    "SVM": SVC(random_state=42, probability=True),
}

predictions = {}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {accuracy:.2f}")
    predictions[name] = preds

# Evaluate on test data
print("\nOpening the evaluation files...\n")
eval_files = [f for f in os.listdir(path_to_eval_tweets)]

# Listes pour stocker les prédictions
dummy_predictions = []
logreg_predictions = []
dectree_predictions = []
randfor_predictions = []
gradboost_predictions = []
xgboost_predictions = []
svm_predictions = []

def majority_vote(group):
    """Compute the majority vote for a group of predictions"""
    return mode(group).mode[0]

for fname in eval_files:
    print(f"Processing {fname}...")
    
    # Lire les données d'évaluation
    val_df = pl.read_csv(f"{path_to_eval_tweets}/{fname}")
    val_df = val_df.with_column(
        pl.col("Tweet").apply(preprocess_text).alias("Tweet")
    )
    
    # Créer les vecteurs d'embedding
    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, 300) for tweet in val_df["Tweet"].to_list()])
    tweet_df = pl.DataFrame(tweet_vectors)
    
    # Combiner les vecteurs avec les données
    val_df = pl.concat([val_df, tweet_df], how="horizontal").drop(["Timestamp", "Tweet"])

    # Prédictions pour chaque modèle
    for name, clf in classifiers.items():
        preds = clf.predict(val_df.drop(["MatchID", "PeriodID", "ID"]).to_numpy())
        val_df = val_df.with_column(
            pl.Series(preds, name=f"{name}EventType")
        )
    
    # Calculer les votes majoritaires par groupe
    dummy_majority = val_df.groupby(["MatchID", "PeriodID", "ID"]).agg(
        [pl.col("DummyEventType").apply(majority_vote).alias("DummyEventType")]
    )
    logreg_majority = val_df.groupby(["MatchID", "PeriodID", "ID"]).agg(
        [pl.col("LogisticRegressionEventType").apply(majority_vote).alias("LogisticRegressionEventType")]
    )
    dectree_majority = val_df.groupby(["MatchID", "PeriodID", "ID"]).agg(
        [pl.col("DecisionTreeEventType").apply(majority_vote).alias("DecisionTreeEventType")]
    )
    randfor_majority = val_df.groupby(["MatchID", "PeriodID", "ID"]).agg(
        [pl.col("RandomForestEventType").apply(majority_vote).alias("RandomForestEventType")]
    )
    gradboost_majority = val_df.groupby(["MatchID", "PeriodID", "ID"]).agg(
        [pl.col("GradientBoostingEventType").apply(majority_vote).alias("GradientBoostingEventType")]
    )
    xgboost_majority = val_df.groupby(["MatchID", "PeriodID", "ID"]).agg(
        [pl.col("XGBoostEventType").apply(majority_vote).alias("XGBoostEventType")]
    )
    svm_majority = val_df.groupby(["MatchID", "PeriodID", "ID"]).agg(
        [pl.col("SVMEventType").apply(majority_vote).alias("SVMEventType")]
    )
    
    # Ajouter les résultats aux listes
    dummy_predictions.append(dummy_majority)
    logreg_predictions.append(logreg_majority)
    dectree_predictions.append(dectree_majority)
    randfor_predictions.append(randfor_majority)
    gradboost_predictions.append(gradboost_majority)
    xgboost_predictions.append(xgboost_majority)
    svm_predictions.append(svm_majority)

# Concaténer et sauvegarder les prédictions
print("Saving predictions...")

pl.concat(dummy_predictions).write_csv("dummy_majority_predictions.csv", has_header=True)
pl.concat(logreg_predictions).write_csv("logistic_majority_predictions.csv", has_header=True)
pl.concat(dectree_predictions).write_csv("decision_tree_majority_predictions.csv", has_header=True)
pl.concat(randfor_predictions).write_csv("random_forest_majority_predictions.csv", has_header=True)
pl.concat(gradboost_predictions).write_csv("gradient_boosting_majority_predictions.csv", has_header=True)
pl.concat(xgboost_predictions).write_csv("xgboost_majority_predictions.csv", has_header=True)
pl.concat(svm_predictions).write_csv("svm_majority_predictions.csv", has_header=True)

print("Majority voting and prediction saving complete.")