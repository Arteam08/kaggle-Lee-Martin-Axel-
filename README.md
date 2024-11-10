# kaggle-Lee-Martin-Axel

*Authors : Axel Delaval, Martin Drieux, Lee Kadz*

## Project: Sub-Event Detection in Football Matches

This project is part of a machine learning challenge to build a binary classification model that predicts notable events (like goals or red cards) during football matches, based on tweet data from the 2010 and 2014 FIFA World Cups. The main objective is to classify each one-minute interval of a match, using tweets, as containing a notable event or not.

### Project Structure

- `./notebook_axel.ipynb`, `./notebook_lee.ipynb`, `./notebook_martin.ipynb`: Jupyter notebooks where team members develop and experiment with different models and preprocessing techniques.
- `./clean_notebook.ipynb` : Jupyter notebooks with the final results.
- `./README.md`: This document.
- `./challenge_data/` : folder with everything which was initially given.

### Data Files

The training and evaluation data includes:
- `train_tweets/*.json`: Annotated tweets per match, with labels indicating notable events.
- `eval_tweets/*.json`: Tweets for evaluation, requiring predictions.

### Getting Started

#### 1. Set Up the Repository

Clone the repository:
```bash
git clone https://github.com/Arteam08/kaggle-Lee-Martin-Axel-.git
cd kaggle-Lee-Martin-Axel-
```
or 
```bash
mkdir kaggle-Lee-Martin-Axel-
cd kaggle-Lee-Martin-Axel-
git init
git remote add origin https://github.com/Arteam08/kaggle-Lee-Martin-Axel-.git
git pull origin main # or master
```

#### 2. Working with Git Commands

- **Add changes to staging**:
  ```bash
  git add <file_name>  # or use '.' to add all changes
  ```

- **Commit changes with a message**:
  ```bash
  git commit -m "Your commit message here"
  ```

- **Push changes to GitHub**:
  ```bash
  git push origin master  # Ensure you're on the correct branch (master or main)
  ```

- **Pull updates from GitHub**:
  ```bash
  git pull origin master  # Sync local branch with the remote repository
  ```

- **Revert files added to staging**:
  ```bash
  git reset <file_name>
  ```

#### 3. Configuring Git for the Project

To ensure consistent handling of branches:
```bash
git config pull.rebase false  # Use 'merge' strategy when pulling updates
```

For more details, refer to the [GitHub CLI documentation](https://docs.github.com/en/github-cli).