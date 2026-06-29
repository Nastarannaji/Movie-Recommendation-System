import numpy as np 
import pandas as pd 
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")

#explore the data 
print(movies.head())

print(movies.info())

print(movies.columns)

print(credits.head())

print(credits.info())

print(credits.columns)


#merging two tables 

movies = movies.merge(credits, left_on="id", right_on="movie_id")

#drop the unnecessary

movies = movies.drop(columns=[
    'budget',
    'homepage',
    'original_language',
    'original_title',
    'popularity',
    'production_companies',
    'production_countries',
    'release_date',
    'revenue',
    'runtime',
    'spoken_languages',
    'status',
    'tagline',
    'vote_average',
    'vote_count'
])

#drop the duplicates
movies = movies.drop(columns=['title_y'])
movies = movies.rename(columns={'title_x': 'title'})

movies.drop(columns=['id'], inplace=True)

print(movies.info())


###changing the feature

#function make the obj into list 
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)

def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].fillna('')

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] + movies['overview']

#keep what we need 
new_df = movies[['movie_id', 'title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


#transform word into text
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

#The similarity score
similarity = cosine_similarity(vectors)

#recommend function 
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]

    distance = similarity[movie_index]

    movies_list = list(enumerate(distance))

    movies_list = sorted(
        movies_list,
        reverse=True,
        key=lambda x: x[1]
    )

    for i in movies_list[1:6]:
        print(new_df.iloc[i[0]].title)


#user input
movie = input("Enter a movie name: ")
recommend(movie)