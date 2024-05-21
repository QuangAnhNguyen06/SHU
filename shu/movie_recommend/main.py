import movieDataset
import contentBase
import collaborative




if __name__ == "__main__":

    df_movies,_,_ = movieDataset.create_df()
    

    # test function 1
    # content-base find movie

    # movieID = int(input('Movie ID:'))
    # b = contentBase.similar_movie(movieID,5)
    # print(f'Movies similar with {movieDataset.get_movie_title(df_movies,movieID)} is {b}')


    # test function 2
    # content-based recommendation
    # userID = int(input('User ID:'))
    # a = contentBase.recommend(userID,5)
    # print(f'Movies recommend to userID {userID} is {a}')



    # test function 3
    # collaborative recommendation
    userID = int(input('User ID:'))
    a = collaborative.user_base(userID,k=5,top_k=5)
    print(f'Movies recommend to userID {userID} is {a}')









    

    
    