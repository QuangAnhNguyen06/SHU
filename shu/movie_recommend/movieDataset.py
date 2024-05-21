import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# data preprocessing

def create_df():

    df_movies = pd.read_csv('movies.csv')
    df_ratings = pd.read_csv('ratings.csv')
    df_ratings = df_ratings.drop(columns=['timestamp'])


    # Tach dataset o cot genre
    df_movies_2 = (df_movies.set_index(['movieId', 'title'])
        .genres.str.split('|', expand=True)
        .stack()
        .reset_index(name='genre')
        .rename(columns={'level_2': 'index'})
        .drop('index', axis=1))

    return df_movies,df_ratings,df_movies_2




# Mot so cac function setup can thiet


def get_user_top_ratings(user_id, dataframe):
    # Lọc dữ liệu theo userId
    user_data = dataframe[dataframe['userId'] == user_id]
    # Giảm dần
    sorted_data = user_data.sort_values(by='rating', ascending=False)
    if len(sorted_data) > 30:
      num_ratings = 15
    if len(sorted_data) <= 30:
      num_ratings = int(len(sorted_data) * 0.3)

    top_ratings = sorted_data.head(num_ratings)

    return top_ratings


def get_top_k_similar_movies(df, k):
    # Sắp xếp DataFrame theo cột similarity_score theo thứ tự giảm dần
    sorted_df = df.sort_values(by='similarity_score', ascending=False)
    # Lấy top k tên bộ phim
    top_k_movie_titles = sorted_df.head(k).index.tolist()
    return top_k_movie_titles



def get_movie_rating_from_user(dataframe, user_id, movie_id):
    user_data = dataframe[dataframe['userId'] == user_id]
    rating = dataframe[dataframe['movieId'] == movie_id]['rating']
    if not rating.empty:
        return rating.values[0]
    else:
        return "No info"


def get_movie_title(dataframe, movie_id):
    movie = dataframe[dataframe['movieId'] == movie_id]['title']
    if len(movie) > 0:
        return movie.values[0]
    else:
        return "No info"


def get_movie_ids(dataframe):
    movie_ids = dataframe['movieId'].values
    return movie_ids



