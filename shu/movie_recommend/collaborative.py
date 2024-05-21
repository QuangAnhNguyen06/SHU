import movieDataset
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import svds
import numpy as np











def SVD():

    df_movies,df_ratings,df_movies_2 = movieDataset.create_df()



    # df dạng   movieID  1   2   3   ...
    #           userID
    #             1
    user_ratings_table = df_ratings.pivot(index='userId', 
                                        columns='movieId', 
                                        values='rating')

    # Get the average rating for each user
    avg_ratings = user_ratings_table.mean(axis=1)


    # Center each users ratings around 0
    user_ratings_table_centered = user_ratings_table.sub(avg_ratings, axis=0)
    # Fill in the missing data with 0s
    user_ratings_table_normed = user_ratings_table_centered.fillna(0)

    #movie_ratings_centered = user_ratings_table_normed.T

    # SVD
    U, sigma, Vt = svds(user_ratings_table_normed.to_numpy())
    S = np.diag(sigma)
    U_sigma = np.dot(U, S)
    # Recreate 
    U_sigma_Vt = np.dot(U_sigma, Vt)

    # Dự đoán xếp hạng mà một người dùng có thể đánh giá một sản phẩm, 
    # ngay cả khi họ chưa bao giờ tương tác với sản phẩm đó trước đây.

    uncentered_ratings = U_sigma_Vt + avg_ratings.values.reshape(-1, 1)

    calc_pred_ratings_df = pd.DataFrame(uncentered_ratings,
                                        index=user_ratings_table.index,
                                        columns=user_ratings_table.columns
                                    )

    return calc_pred_ratings_df

    # ma trận output 668 rows × 10325 columns
    # movieID   1   2   3   4   5   ...
    # userID
    #   1       3.6 3.7 3.4 ... 
    #   2       4.1 3.7 3.8 ... 
    #   3       ... ....






























def user_base(userID, k=5, top_k=5):

    df_movies,df_ratings,df_movies_2 = movieDataset.create_df()



    # df dạng   movieID  1   2   3   ...
    #           userID
    #             1
    user_ratings_table = df_ratings.pivot(index='userId', 
                                        columns='movieId', 
                                        values='rating')

    # Get the average rating for each user
    avg_ratings = user_ratings_table.mean(axis=1)

    # Center each users ratings around 0
    user_ratings_table_centered = user_ratings_table.sub(avg_ratings, axis=0)
    # Fill in the missing data with 0s
    user_ratings_table_normed = user_ratings_table_centered.fillna(0)
    # Tạo xong df


    # Tính tương đồng
    user_similarities = cosine_similarity(user_ratings_table_normed)
    #user_similarities
    cosine_user_similarity_df = pd.DataFrame(user_similarities,
                                            index=user_ratings_table_normed.index,
                                             columns=user_ratings_table_normed.index)
    
    user_similarity_series = cosine_user_similarity_df.loc[userID]
    ordered_user_similarities = user_similarity_series.sort_values(ascending=False)
    nearest_neighbors = ordered_user_similarities[1:k+1].index
    neighbor_ratings = user_ratings_table_normed.reindex(nearest_neighbors)


    #return neighbor_ratings[movieID].mean() + avg_ratings[userID]

    dict_output = {}
    for i in neighbor_ratings.columns:
        dict_output[i] = neighbor_ratings[i].mean() + avg_ratings[userID]

    top_k_keys = [key for key, value in sorted(dict_output.items(), key=lambda item: item[1], reverse=True)[:top_k]]

    return [movieDataset.get_movie_title(df_movies,i) for i in top_k_keys]




