####################################################################################################
####################################################################################################
####################################### import libraries ###########################################
####################################################################################################
####################################################################################################

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


def download_necessary_models():
    # Stopwords
    nltk.download('stopwords')
    # Wordnet
    nltk.download('wordnet')
    # Punkt
    nltk.download('punkt')

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

# Charger le modèle GloVe (50 dimensions)
embeddings_model = api.load("glove-twitter-50")

####################################################################################################
####################################################################################################
####################################### load the dataset ###########################################
####################################################################################################
####################################################################################################

def load_data(path, verbose=False):
    li = []
    for filename in os.listdir(path):
        df = pl.read_csv(path + "/" + filename)
        li.append(df)
    output = pl.concat(li)
    if verbose:
        print(output.head())
        print(f'The shape of the data is: {output.shape}')
    return output

####################################################################################################
####################################################################################################
####################################### preprocess the text ########################################
####################################################################################################
####################################################################################################

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

# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=50):
    """ 
    Compute the average embedding vector for a tweet
    
    Parameters:
        tweet (str): The tweet text
        model (gensim.models.keyedvectors.Word2VecKeyedVectors): The word embeddings model
        vector_size (int): The size of the word embedding vectors
        
    Returns:
        np.ndarray: The average embedding vector for the tweet
    """

    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


# Basic preprocessing function
def preprocess_text(text, 
                    nb_it, 
                    nb_max_it, 
                    with_URLs=True,
                    with_trigrams=True,
                    with_spelling_correction=True,
                    with_emojis=True,
                    with_emoticons=True,
                    with_lowercasing=True,
                    with_punctuation=True,
                    with_numbers=True,
                    with_stopwords=True, 
                    with_lemmatization=True, 
                    with_stemming=True, 
                    verbose=False):
    if verbose and (nb_it % (nb_max_it//100) == 0): 
        print(f"Processing text {nb_it} / {nb_max_it} ({nb_it/nb_max_it*100:.2f}%)")
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

# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=50):
    """ 
    Compute the average embedding vector for a tweet
    
    Parameters:
        tweet (str): The tweet text
        model (gensim.models.keyedvectors.Word2VecKeyedVectors): The word embeddings model
        vector_size (int): The size of the word embedding vectors
        
    Returns:
        np.ndarray: The average embedding vector for the tweet
    """

    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


def embed_tweets(df, model, embeddings_sentence_model, vector_size=50):

    tweet_vectors = np.vstack([
        embeddings_sentence_model(tweet, model, vector_size) for tweet in df["Tweet"].to_list()
    ])

    # Create a DataFrame from tweet vectors
    tweet_df = pl.DataFrame(tweet_vectors)

    # Attach the vectors to the original dataframe
    period_features = pl.concat([df, tweet_df], how="horizontal")

    # Drop the columns that are not useful anymore
    period_features = period_features.drop(["Timestamp", "Tweet"])

    # Group the tweets into their corresponding periods and calculate the mean
    period_features = (
        period_features.group_by(["MatchID", "PeriodID", "ID"])
        .agg(pl.all().mean())
    )

    return period_features

####################################################################################################
####################################################################################################
############################################### NN #################################################
####################################################################################################
####################################################################################################

# Define the Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(FeedforwardNN, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Adjust the Dataset class if necessary
class FeedforwardDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Shape: (num_samples, input_size)
        self.y = torch.tensor(y, dtype=torch.long)     # Shape: (num_samples,)
    
        # Optionally, normalize the data
        # self.X = (self.X - self.X.mean(dim=0)) / self.X.std(dim=0)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Train and evaluate the Neural Network
def train_feedforward_nn(X_train, y_train, X_test, y_test, input_size, hidden_sizes, num_classes, epochs=10, batch_size=64, lr=0.001):
    # Create datasets and loaders
    train_dataset = FeedforwardDataset(X_train, y_train)
    test_dataset = FeedforwardDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedforwardNN(input_size, hidden_sizes, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = correct / total
    print(f"Feedforward Neural Network Accuracy: {accuracy:.4f}")
    return accuracy


####################################################################################################
####################################################################################################
####################################### compute the accuracy #######################################
####################################################################################################
####################################################################################################

def compute_accuracy(path_to_training_tweets, 
                        embeddings_model,
                        keep_only=100000,
                        with_URLs=True,
                        with_trigrams=True,
                        with_spelling_correction=True,
                        with_emojis=True,
                        with_emoticons=True,
                        with_lowercasing=True,
                        with_punctuation=True,
                        with_numbers=True,
                        with_stopwords=True, 
                        with_lemmatization=True, 
                        with_stemming=True, 
                        verbose=False):
    # Load the data
    df = load_data(path_to_training_tweets, verbose = verbose)
    # Keep only the first tweets
    if verbose:
        print(f"Keeping only the first {keep_only} tweets")
    df = df.head(keep_only)
    # Apply preprocessing to each tweet
    tweets_preprocessed = [preprocess_text(tweet, i, len(df["Tweet"]),
                                            with_URLs=with_URLs,
                                            with_trigrams=with_trigrams,
                                            with_spelling_correction=with_spelling_correction,
                                            with_emojis=with_emojis,
                                            with_emoticons=with_emoticons,
                                            with_lowercasing=with_lowercasing,
                                            with_punctuation=with_punctuation,
                                            with_numbers=with_numbers,
                                            with_stopwords=with_stopwords,
                                            with_lemmatization=with_lemmatization,
                                            with_stemming=with_stemming,
                                            verbose=verbose) for i, tweet in enumerate(df["Tweet"].to_list())]

    # Create a new DataFrame with the preprocessed tweets
    period_features = embed_tweets(df, embeddings_model, get_avg_embedding, vector_size=50)

    # Drop non-numerical features and keep embedding values
    X = period_features.drop(["EventType", "MatchID", "PeriodID", "ID"]).to_numpy()
    # Extract labels
    y = period_features["EventType"].to_numpy()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a logistic regression classifier
    clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def compare_models(path_to_training_tweets, embeddings_model, keep_only=100000, 
                   with_URLs=True, with_trigrams=True, with_spelling_correction=True, 
                   with_emojis=True, with_emoticons=True, with_lowercasing=True, 
                   with_punctuation=True, with_numbers=True, with_stopwords=True, 
                   with_lemmatization=True, with_stemming=True, verbose=False):
    # Load the data
    df = load_data(path_to_training_tweets, verbose=verbose)
    length = len(df["Tweet"])
    if keep_only == 'MAX':
        keep_only = length
    # Keep only the first tweets
    if verbose:
        print(f"Keeping only the first {keep_only} tweets")
    df = df.head(keep_only)
    # Apply preprocessing
    tweets_preprocessed = [
        preprocess_text(tweet, i, len(df["Tweet"]),
                        with_URLs=with_URLs,
                        with_trigrams=with_trigrams,
                        with_spelling_correction=with_spelling_correction,
                        with_emojis=with_emojis,
                        with_emoticons=with_emoticons,
                        with_lowercasing=with_lowercasing,
                        with_punctuation=with_punctuation,
                        with_numbers=with_numbers,
                        with_stopwords=with_stopwords,
                        with_lemmatization=with_lemmatization,
                        with_stemming=with_stemming,
                        verbose=verbose) for i, tweet in enumerate(df["Tweet"].to_list())]
    
    # Update the DataFrame with preprocessed tweets
    df = df.with_columns(pl.Series(name="Tweet", values=tweets_preprocessed))
    
    # Embed tweets into vectors
    period_features = embed_tweets(df, embeddings_model, get_avg_embedding, vector_size=50)
    
    # Prepare data
    X = period_features.drop(["EventType", "MatchID", "PeriodID", "ID"]).to_numpy()
    y = period_features["EventType"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Models to test
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        "SVM": SVC(random_state=42, probability=True)
    }
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name}: Accuracy = {accuracy:.4f}")
    
    # # Add Neural Network evaluation
    # print("\nTesting Neural Network...")
    # nn_accuracy = train_feedforward_nn(
    #     X_train, y_train, X_test, y_test,
    #     input_size=X_train.shape[1],
    #     hidden_sizes=[128, 64],  # Example hidden layer sizes
    #     num_classes=len(np.unique(y)),  # Number of classes
    #     epochs=10,
    #     batch_size=64,
    #     lr=0.001
    # )
    # results["Feedforward Neural Network"] = nn_accuracy

    return results

####################################################################################################
####################################################################################################
############################################# main #################################################
####################################################################################################
####################################################################################################

if __name__ == "__main__":
    verbose = True
    # Download necessary models
    # download_necessary_models()
    # Compare models
    results = compare_models(path_to_training_tweets, embeddings_model=embeddings_model, keep_only='MAX', verbose=verbose, with_spelling_correction=False)
    print("\nModel Comparison Results:")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")