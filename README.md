# AnimeVerse-DataAnalysis
Anime is a popular form of entertainment originating from Japan, characterized by vibrant artwork, fantastical themes, and colorful characters. This project leverages the Anime dataset to perform various data-driven analyses, helping producers and streaming platforms make strategic decisions about future anime productions.

## Project Overview
This project explores anime popularity and viewer sentiments using data analysis techniques like genre analysis, sentiment analysis, and machine learning models. We focus on understanding how factors like genre, episode count, and viewer ratings influence anime success. Techniques such as Random Forest, LDA, and TF-IDF are used to provide actionable insights for producers and streaming platforms.

## Dataset
The Anime Dataset used in this project contains data from 1963 to 2023, featuring key attributes like titles, genres, airing dates, number of episodes, popularity, and user scores.

**Key features of the dataset:**

**Title:** The name of the anime
**Genres:** Categories like Action, Comedy, etc.
**Episodes:** The number of episodes aired
**Score:** User rating of the anime
**Popularity:** Popularity rank based on community votes

## Code Explaination

#### Installing Libraries and Data Loading:
```{python}
import pandas as pd
import matplotlib.pyplot as plt

data_path = "C:/Users/rajpu/Downloads/animes_dataset.csv"
anime_df = pd.read_csv(data_path)


```
#### Exploratory Data Analysis and Preprocessing:
```{python}
# Display the first few rows of the dataset
anime_df.head()

# Check for any missing values in the dataset
missing_values = anime_df.isnull().sum()

# Fill missing values in relevant columns
anime_df['genre'] = anime_df['genre'].fillna('Unknown')



```

#### Data Visualization:
```{python}
# Plot the distribution of anime genres
anime_df['genre'].value_counts().plot(kind='bar', figsize=(10,5))
plt.title("Distribution of Anime Genres")
plt.xlabel("Genres")
plt.ylabel("Count")
plt.show()




```
