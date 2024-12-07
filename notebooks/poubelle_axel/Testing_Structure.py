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
# embeddings_model = api.load("glove-twitter-50")
# embeddings_model = api.load("glove-wiki-gigaword-300")
embeddings_model = api.load("fasttext-wiki-news-subwords-300")
# embeddings_model = api.load("word2vec-google-news-300")


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

# Define a custom Dataset class for PyTorch
class TweetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert features to tensors
        self.y = torch.tensor(y, dtype=torch.long)     # Convert labels to tensors

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNet, self).__init__()
        # Create a list of layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
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
                   with_lemmatization=True, with_stemming=True, verbose=False, vector_size=50):
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

    ## save into a csv file 
    # df.write_csv('train_tweets_preprocessed.csv')
    
    # Embed tweets into vectors
    period_features = embed_tweets(df, embeddings_model, get_avg_embedding, vector_size=vector_size)
    
    # Prepare data
    X = period_features.drop(["EventType", "MatchID", "PeriodID", "ID"]).to_numpy()
    y = period_features["EventType"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Models to testx
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

    # PyTorch Neural Network
    print("Training PyTorch Neural Network...")
    # Define training parameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001
    input_size = X_train.shape[1]
    hidden_sizes = [128, 64]  # Two hidden layers with 128 and 64 neurons
    num_classes = len(np.unique(y_train))

    # Create datasets and dataloaders
    train_dataset = TweetDataset(X_train, y_train)
    test_dataset = TweetDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = NeuralNet(input_size, hidden_sizes, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        results['PyTorch Neural Network'] = accuracy
        print(f'PyTorch Neural Network: Accuracy = {accuracy:.4f}')
    
    return results

def compare_models_from_csv(path_to_training_tweets_csv, embeddings_model, vector_size=50):
    # Load the data
    df = pl.read_csv(path_to_training_tweets_csv)
    
    # Embed tweets into vectors
    period_features = embed_tweets(df, embeddings_model, get_avg_embedding, vector_size=vector_size)
    
    # Prepare data
    X = period_features.drop(["EventType", "MatchID", "PeriodID", "ID"]).to_numpy()
    y = period_features["EventType"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Models to testx
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

    # PyTorch Neural Network
    print("Training PyTorch Neural Network...")
    # Define training parameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001
    input_size = X_train.shape[1]
    hidden_sizes = [128, 64]  # Two hidden layers with 128 and 64 neurons
    num_classes = len(np.unique(y_train))

    # Create datasets and dataloaders
    train_dataset = TweetDataset(X_train, y_train)
    test_dataset = TweetDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = NeuralNet(input_size, hidden_sizes, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        results['PyTorch Neural Network'] = accuracy
        print(f'PyTorch Neural Network: Accuracy = {accuracy:.4f}')
    
    return results, models

def submission(
    model, 
    path_to_training_tweets_csv, 
    path_to_eval_tweets, 
    embeddings_model, 
    vector_size=50,
    output_file="submission.csv",
    verbose=False,
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
    with_stemming=True
):
    """
    Generate predictions, display model accuracy, and save them in a CSV file for submission.
    """
    # Load preprocessed training data
    train_df = pl.read_csv(path_to_training_tweets_csv)

    # Embed tweets into vectors for training
    period_features = embed_tweets(train_df, embeddings_model, get_avg_embedding, vector_size=vector_size)

    # Extract features and labels
    X_train = period_features.select(pl.exclude(["EventType", "MatchID", "PeriodID", "ID"])).to_numpy()
    y_train = period_features["EventType"].to_numpy()

    # Split data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Train the model on the training split
    print("Training the model on the training split...")
    model.fit(X_train_split, y_train_split)

    # Predict and calculate accuracy on the validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Approximate validation accuracy: {val_accuracy:.4f}")

    # Train the model on the full training data
    print("Training the model on the full training data...")
    model.fit(X_train, y_train)

    # Prepare predictions for the evaluation set
    predictions = []
    for fname in os.listdir(path_to_eval_tweets):
        try:
            eval_df = pd.read_csv(os.path.join(path_to_eval_tweets, fname))

            # Preprocess tweets
            eval_df['Tweet'] = eval_df['Tweet'].apply(lambda tweet: preprocess_text(
                tweet,
                nb_it=0,
                nb_max_it=1,
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
                with_stemming=with_stemming
            ))

            # Generate embeddings for evaluation data
            tweet_vectors = np.vstack([
                get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in eval_df['Tweet']
            ])
            tweet_df = pd.DataFrame(tweet_vectors, columns=[f"dim_{i}" for i in range(vector_size)])

            # Combine and aggregate features
            eval_features = pd.concat([eval_df, tweet_df], axis=1)
            eval_features = eval_features.drop(columns=['Timestamp', 'Tweet'])
            eval_features = eval_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

            # Ensure the feature columns match the training data
            X_eval = eval_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

            # Check if feature dimensions match
            if X_eval.shape[1] != X_train.shape[1]:
                raise ValueError(f"Feature shape mismatch, expected: {X_train.shape[1]}, got {X_eval.shape[1]}")

            # Predict event types
            preds = model.predict(X_eval)

            # Save predictions
            eval_features['EventType'] = preds
            predictions.append(eval_features[['ID', 'EventType']])

        except Exception as e:
            print(f"Error processing file {fname}: {e}")
            continue

    # Check if predictions list is not empty
    if predictions:
        # Concatenate all predictions
        submission_df = pd.concat(predictions, axis=0)
        # Save predictions to a CSV file
        submission_df.to_csv(output_file, index=False)
        print(f"Submission file saved to: {output_file}")
    else:
        print("No predictions were generated. Please check the feature preparation steps.")

    return submission_df if predictions else None


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
    results = compare_models(path_to_training_tweets, embeddings_model=embeddings_model, keep_only=600000, verbose=verbose, with_spelling_correction=False, vector_size=300)
    print("\nModel Comparison Results:")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")

    # Compare models from csv
    # results, models  = compare_models_from_csv("train_tweets_preprocessed.csv", embeddings_model=embeddings_model, vector_size=50)
    # print("\nModel Comparison Results:")
    # for model_name, accuracy in results.items():
    #     print(f"{model_name}: {accuracy:.4f}")

    # model = XGBClassifier(random_state=42, eval_metric='mlogloss')
    # model = RandomForestClassifier(random_state=42, n_estimators=100)

    # Submission
    # submission(model, "train_tweets_preprocessed.csv", path_to_eval_tweets, embeddings_model=embeddings_model, verbose=verbose, with_spelling_correction=False)
