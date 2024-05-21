import main 
import streamlit as st
import contentBase
import movieDataset
import collaborative






def sum_numbers(num1, num2):
    return num1 + num2

def subtract_numbers(num1, num2):
    return num1 - num2


def multiply_numbers(num1, num2):
    return num1 * num2

def divide_numbers(num1, num2):
    if num2 != 0:
        return num1 / num2
    else:
        return "Không thể chia cho 0"




# Giao diện Streamlit
def main():
    st.title("Movie Recommendation")

    # Nút 
    if st.button("Movie tương đồng"):
        num1 = st.number_input("Nhập MovieID:", value=0)
        #num2 = st.number_input("Nhập số thứ hai:", value=0)
        #userID = num2
        
        movieID = num1
        result = contentBase.recommend(movieID,5)

        st.write("Kết quả là:", result)

    #
    if st.button("Tính Hiệu"):
        num1 = st.number_input("Nhập MovieID:", value=0)
        userID = num1

        result = contentBase.similar_movie(userID,5)

        st.write("Kết quả là:", result)





if __name__ == "__main__":
    main()
