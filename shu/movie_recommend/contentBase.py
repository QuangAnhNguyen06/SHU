import movieDataset
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform





# Tìm các movie giống với 1 movieID cho vào
def similar_movie(movieID = 1, k=5):
    df_movies,df_ratings,df_movies_2 = movieDataset.create_df()
    
    movie_genre_cross_table = pd.crosstab(df_movies_2['title'],
                                           df_movies_2['genre'])    

# Cosine
# Tính sự tương đồng các movie dựa theo genre và sử dụng Jaccard
    # Calculate all pairwise distances
    #jaccard_distances = pdist(movie_genre_cross_table.values, metric='jaccard')

    # Convert the distances to a square matrix
    #jaccard_similarity_array = 1 -  squareform(jaccard_distances)
    
    
    # Cosine similarity\
    # (jaccard_similarity_df) - use this var name
    cosine_similarity_df = pd.DataFrame(cosine_similarity(movie_genre_cross_table), 
                                        index=movie_genre_cross_table.index,
                                        columns=movie_genre_cross_table.index)


    # Wrap the array in a pandas DataFrame
    # jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, 
    #                                      index=movie_genre_cross_table.index, 
    #                                      columns=movie_genre_cross_table.index)
    

    def top_similar_movies(movie_title, similarity_df, k):
        # Lấy dòng tương ứng với bộ phim có tên movie_title
        movie_row = similarity_df.loc[movie_title]
        
        # Sắp xếp các điểm số độ tương tự và lấy ra top k+1 (bao gồm chính bộ phim đó)
        top_similarities = movie_row.sort_values(ascending=False)[1:k+1]
        
        return top_similarities

    movie_title = movieDataset.get_movie_title(df_movies,movieID)
    top_k_similar_movies_df = top_similar_movies(movie_title, cosine_similarity_df, k)
    top_k_similar_movies = top_k_similar_movies_df.head(k).index.tolist()
    return(top_k_similar_movies)    









# cho 1 userID, tìm các movie đề xuất cho id đó dựa theo top rating
def recommend(id, k = 10):

    df_movies,df_ratings,df_movies_2 = movieDataset.create_df()
    #id = 1 # userID cần lấy
    #k = 10 # số kết quả muốn nhận lại

    movie_genre_cross_table = pd.crosstab(df_movies_2['title'],
                                           df_movies_2['genre'])


    # JACCARD 
# # Tính sự tương đồng các movie dựa theo genre và sử dụng Jaccard
#     # Calculate all pairwise distances
#     jaccard_distances = pdist(movie_genre_cross_table.values, metric='jaccard')

#     # Convert the distances to a square matrix
#     jaccard_similarity_array = 1 -  squareform(jaccard_distances)

#     # Wrap the array in a pandas DataFrame
#     jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, 
#                                          index=movie_genre_cross_table.index, 
#                                          columns=movie_genre_cross_table.index)
    

    # Cosine similarity\
    # (jaccard_similarity_df) - use this var name
    cosine_similarity_df = pd.DataFrame(cosine_similarity(movie_genre_cross_table), 
                                        index=movie_genre_cross_table.index,
                                        columns=movie_genre_cross_table.index)






# Lấy thông tin top rating của user
    df_top_rate_movie = movieDataset.get_user_top_ratings(id,df_ratings)
    #list_of_id_movies_enjoyed
    list_of_id_movies_enjoyed = movieDataset.get_movie_ids(df_top_rate_movie)
    list_of_name_movies_enjoyed = [movieDataset.get_movie_title(df_movies, i) for i in list_of_id_movies_enjoyed]



# Tạo ra các subset
    cosine_similarity_df_2 = cosine_similarity_df.copy()
    
    # Shape: 10312 rows × 10327 columns
    subset_df = cosine_similarity_df_2.drop(list_of_name_movies_enjoyed, axis=0)


# Tạo user profile bằng cách dựa theo các movie top rating đã tính, tính sự tương 
# đồng của chúng đến tất cả các movie khác   
# Shape: 15 rows × 10327 columns
# Sau đó tính mean và tạo ra profile user 1 x 10327
    subset_df_2 = cosine_similarity_df_2.reindex(list_of_name_movies_enjoyed)
    user_profile = subset_df_2.mean()



# Tính sự tương đồng cosine giữa các movie chưa xem và profile của user
# user_profile.values.reshape(1, -1) : (1, 10327)
# subset_df: 10312 rows × 10327 columns chứa tương đồng jaccard giữa các phim chưa xem
# và tất cả các phim khác, các feature sử dụng chính là các cột tương ứng các phim

    # Calculate the cosine_similarity and wrap it in a DataFrame
    similarity_array = cosine_similarity(user_profile.values.reshape(1, -1), 
                                         subset_df)
    
    similarity_df = pd.DataFrame(similarity_array.T, 
                                 index=subset_df.index, 
                                 columns=["similarity_score"])

    # Sort the values from high to low by the values in the similarity_score
    sorted_similarity_df = similarity_df.sort_values(by="similarity_score", 
                                                     ascending=False)


# Lấy top kết quả
    # Sắp xếp DataFrame theo cột similarity_score theo thứ tự giảm dần
    sorted_df = sorted_similarity_df.sort_values(by='similarity_score', ascending=False)
    # Lấy top k tên bộ phim
    top_k_movie_titles = sorted_df.head(k).index.tolist()
    return (top_k_movie_titles)




