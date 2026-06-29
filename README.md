# Movie-Recommendation-System
A machine learning movie recommendation system that suggests similar movies using content-based filtering and natural language processing.
# 🎬 Movie Recommendation System

## Overview

This project is a **content-based movie recommendation system** built with Python and machine learning techniques. Given the title of a movie, the system recommends similar movies based on their content, such as genres, keywords, cast, director, and plot overview.

The recommendation engine uses **Natural Language Processing (NLP)** and **Cosine Similarity** to compare movies and find the most relevant recommendations.

---

## Features

* Merge and clean movie datasets
* Extract important movie information
* Create a combined feature (`tags`) for each movie
* Convert text data into numerical vectors using **CountVectorizer**
* Calculate movie similarity using **Cosine Similarity**
* Recommend the top 5 most similar movies based on user input

---

## Dataset

This project uses the **TMDB 5000 Movie Dataset**, which contains information about 5,000 movies including:

* Genres
* Keywords
* Cast
* Crew
* Movie overview

The dataset is available on Kaggle.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CountVectorizer
* Cosine Similarity
* AST (for parsing JSON-like strings)

---

## Project Workflow

1. Load the movie and credits datasets.
2. Merge both datasets into a single DataFrame.
3. Remove unnecessary columns.
4. Clean and extract useful information:

   * Genres
   * Keywords
   * Cast (top 3 actors)
   * Director
   * Overview
5. Combine all features into a single `tags` column.
6. Convert the text into numerical vectors using **CountVectorizer**.
7. Compute **Cosine Similarity** between every movie.
8. Allow the user to enter a movie title and receive the top 5 recommendations.

---

## Example

**Input**

```
Enter a movie:
Avatar
```

**Output**

```
Recommended Movies:

1. Guardians of the Galaxy
2. John Carter
3. Star Trek
4. Aliens
5. Titan A.E.
```

---

## Project Structure

```
movie-recommendation/
│
├── movie-recom.py
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
└── README.md
```

---

## How to Run

1. Clone the repository.

2. Install the required libraries:

```
pip install pandas numpy scikit-learn
```

3. Run the project:

```
python movie-recom.py
```

4. Enter the name of a movie and receive recommendations.

---

## Skills Demonstrated

* Data Cleaning
* Feature Engineering
* Natural Language Processing (NLP)
* Machine Learning
* Recommendation Systems
* Python Programming
* Pandas Data Manipulation
* Text Vectorization
* Cosine Similarity

---

## Future Improvements

* Build a graphical user interface (GUI)
* Create a web application using Flask or Streamlit
* Add movie posters using the TMDB API
* Improve recommendations using collaborative filtering
* Deploy the project online

---

## Author

**Nastaran Naji**

This project was built as part of my Data Science and Machine Learning learning journey.
