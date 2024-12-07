import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import  os
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline


def load_data(path, verbose=False):
    "load un data set avec plusieurs fichiers csv"
    li = []
    for filename in os.listdir(path):
        df = pl.read_csv(path + "/" + filename)
        li.append(df)
    output = pl.concat(li)
    if verbose:
        print(output.head())
        print(f'The shape of the data is: {output.shape}')
    return output


def add_time_sec(df):
    '''Ajoute une colonne "Seconds" représentant le temps écoulé en secondes depuis le début de chaque match 
       et supprime la colonne "Timestamp" à un df polar.'''
    
    # Calcul des timestamps initiaux par match
    first_timestamps = df.group_by("MatchID").agg(
        pl.col("Timestamp").min().alias("FirstTimestamp")
    )
    
    # Jointure pour récupérer le premier timestamp de chaque match
    df = df.join(first_timestamps, on="MatchID")
    
    # Calcul des secondes écoulées
    df = df.with_columns(
        ((pl.col("Timestamp") - pl.col("FirstTimestamp")) // 1000).alias("Seconds")
    )
    
    # Suppression des colonnes inutiles
    df = df.drop(["Timestamp", "FirstTimestamp"])
    
    return df


def df_freq_normalized(df):
    '''Crée un dataframe polar qui contient pour chaque seconde le nombre de tweets normalisé par le nombre moyen de tweets par seconde,
       ainsi que la moyenne de la colonne EventType, tout en conservant la colonne PeriodID.'''
    
    # Calcul des effectifs par seconde et par match
    effectifs_per_sec = df.group_by(["MatchID", "Seconds", "PeriodID"]).agg(
        pl.count("MatchID").alias("Effectif")
    )
    
    # Calcul de la durée totale par match
    match_durations = df.group_by(["MatchID", "PeriodID"]).agg(
        (pl.col("Seconds").max() - pl.col("Seconds").min()).alias("MatchDuration")
    )
    
    # Calcul de la moyenne de EventType par seconde et par match
    eventtype_means = df.group_by(["MatchID", "Seconds", "PeriodID"]).agg(
        pl.col("EventType").mean().alias("EventType_Mean")
    )
    
    # Jointure des durées avec les effectifs
    effectifs_per_sec = effectifs_per_sec.join(match_durations, on=["MatchID", "PeriodID"])
    
    # Jointure des moyennes de EventType
    effectifs_per_sec = effectifs_per_sec.join(eventtype_means, on=["MatchID", "Seconds", "PeriodID"])
    
    # Normalisation des effectifs par match, puis multiplication par la durée
    effectifs_per_sec = effectifs_per_sec.with_columns(
        ((pl.col("Effectif") / pl.col("Effectif").sum().over("MatchID")) * pl.col("MatchDuration"))
        .alias("Effectif_Normalisé")
    )
    
    return effectifs_per_sec


def gaussian_filter(array, sigma=50):
    """
    Applique un filtre gaussien à un array 1D.
    :param array: Array 1D d'entrée.
    :param sigma: Écart-type du filtre gaussien.
    :return: Array 1D filtré.
    """
    return gaussian_filter1d(array, sigma)




def derivee_gaussian_abs(array, sigma=50):
    """
    Calcule la dérivée discrète d'un signal bruité après un flou gaussien.
    :param signal: Array 1D du signal d'entrée.
    :param delta_t: Intervalle de temps entre deux points.
    :param sigma: Écart-type du flou gaussien.
    :return: Array 1D représentant la dérivée lissée.
    """
    # Appliquer le flou gaussien

    signal_lissé = gaussian_filter1d(array, sigma=50)

    # Calcul des différences centrées pour la dérivée
    n = len(array)
    derivative = np.empty(n)

    if n > 2:
        # Différence centrée pour les points internes
        derivative[1:-1] = (signal_lissé[2:] - signal_lissé[:-2]) / 2

        # Différences unilatérales pour les bords
        derivative[0] = (signal_lissé[1] - signal_lissé[0]) 
        derivative[-1] = (signal_lissé[-1] - signal_lissé[-2]) 
    elif n == 2:
        derivative[0] = (signal_lissé[1] - signal_lissé[0])
        derivative[1] = (signal_lissé[1] - signal_lissé[0])
    else:
        derivative[0] = 0

    return np.abs(derivative)


def derivee_seg_gaussian(array, sigma=50):
    """
    Calcule la dérivée seconde d'un signal bruité après un flou gaussien.
    :param array: Array 1D du signal d'entrée.
    :param Lsigma: Liste contenant l'écart-type du flou gaussien (sigma).
    :return: Array 1D représentant la dérivée seconde gaussienne.
    """

    n=len(array)

    # Appliquer le flou gaussien pour lisser le signal
    signal_lissé = gaussian_filter1d(array, sigma)
    second_derivative = np.empty(n)
    if n > 2:
        # Différence centrée pour les points internes
        second_derivative[1:-1] = (signal_lissé[2:] - 2 * signal_lissé[1:-1] + signal_lissé[:-2])

        # Différences unilatérales pour les bords
        second_derivative[0] = (signal_lissé[2] - 2 * signal_lissé[1] + signal_lissé[0])
        second_derivative[-1] = (signal_lissé[-1] - 2 * signal_lissé[-2] + signal_lissé[-3])
    elif n == 2:
        second_derivative[0] = (signal_lissé[1] - 2 * signal_lissé[0] + signal_lissé[1])
        second_derivative[1] = (signal_lissé[1] - 2 * signal_lissé[0] + signal_lissé[1])
    else:
        second_derivative[0] = 0
        second_derivative[0] = 0  # or some other appropriate value

    if n > 3:
        second_derivative[-1] = (signal_lissé[-1] - 2 * signal_lissé[-2] + signal_lissé[-3])
    else:
        second_derivative[-1] = 0  # or some other appropriate value

    return np.abs(second_derivative)


def df_sec_new_features(df_sec, sigma):
    '''Crée un dataframe polar qui contient les nouvelles features dérivées des effectifs normalisés par seconde.'''
    
    # Conversion de la colonne "Effectif_Normalisé" en array NumPy
    effectif_array = df_sec["Effectif_Normalisé"].to_numpy()

    # Application des transformations
    effectif_lissé = gaussian_filter(effectif_array, sigma)
    effectif_derivé = derivee_gaussian_abs(effectif_array, sigma)
    effectif_derivé_seg = derivee_seg_gaussian(effectif_array, sigma)

    # Ajout des colonnes au dataframe Polars
    df_sec = df_sec.with_columns(
        pl.Series("Effectif_Normalisé_Lissé", effectif_lissé),
        pl.Series("Effectif_Normalisé_Lissé_Derivé", effectif_derivé),
        pl.Series("Effectif_Normalisé_Lissé_Derivé_Seg", effectif_derivé_seg)
    )
    
    return df_sec


def split_sets(df, match_ids):
    df_test = df.filter(pl.col("MatchID").is_in(match_ids))
    df_train = df.filter(~pl.col("MatchID").is_in(match_ids))
    return df_train, df_test


def vote_by_period(df, labels, threshold=0.5):
    '''Crée un dataframe polar qui contient les votes majoritaires par période.'''
    
    # Calcul des votes majoritaires par seconde
    votes_per_sec = df.group_by(["MatchID", "Seconds"]).agg(
        pl.col("Vote").max().alias("Vote")
    )
    
    # Calcul des votes majoritaires par période
    votes_per_period = votes_per_sec.group_by("MatchID").agg(
        pl.col("Vote").mean().alias("Vote")
    )
    
    # Seuillage des votes
    votes_per_period = votes_per_period.with_columns(
        pl.when(pl.col("Vote") > threshold, 1).otherwise(0).alias("Vote")
    )
    
    # Ajout des labels
    votes_per_period = votes_per_period.with_columns(
        pl.Series("Label", labels)
    )
    
    return votes_per_period

def aggregate_labels_with_vote(df, predicted_labels, threshold=0.5, group_keys=["MatchID", "PeriodID"], target_column="EventType_Mean"):
    """
    Aggregates true labels and applies a voting mechanism on predicted labels,
    returning both as NumPy arrays.

    :param df: DataFrame Polars initial containing the data and true labels.
    :param predicted_labels: Array of predicted labels corresponding to the rows in df.
    :param threshold: Threshold for voting mechanism on predicted labels.
    :param group_keys: List of columns used for grouping (default: ["MatchID", "PeriodID"]).
    :param target_column: Name of the column containing true labels.

    :return: Tuple (aggregated true labels as array, aggregated predicted labels as array).
    """
    import polars as pl
    import numpy as np

    # Debugging: Print initial columns
    print("Initial columns in df:", df.columns)

    # Add predicted labels as a new column
    df = df.with_columns(pl.Series(name="Prediction", values=predicted_labels))

    # Debugging: Print columns after adding predictions
    print("Columns after adding Prediction:", df.columns)

    # Group and aggregate true labels
    true_labels_grouped = df.group_by(group_keys).agg(
        pl.col(target_column).mean().alias("TrueLabelAggregated")
    )

    # Group and apply voting on predictions
    predicted_labels_grouped = df.group_by(group_keys).agg(
        (pl.col("Prediction").mean() >= threshold).alias("PredictedLabelAggregated")
    )

    # Extract the aggregated labels as NumPy arrays
    true_labels_array = true_labels_grouped["TrueLabelAggregated"].to_numpy()
    predicted_labels_array = predicted_labels_grouped["PredictedLabelAggregated"].to_numpy()

    return true_labels_array, predicted_labels_array



