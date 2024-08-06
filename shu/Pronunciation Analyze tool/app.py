import streamlit as st
from streamlit import session_state
import task1
import get_dict

# Đặt tiêu đề cho trang chính
st.title("Main Page")

# Tạo các nút để chuyển sang các trang khác
st.subheader("Choose a function to proceed:")

# Kiểm tra nếu session_state chưa có thuộc tính page
if 'page' not in session_state:
    session_state.page = "main"

# Điều hướng dựa trên giá trị của session_state.page
if session_state.page == "main":
    col1, col2 = st.columns(2)

    with col1:
        if st.button('Comparison'):
            session_state.page = "task1"
            st.experimental_rerun()

    with col2:
        if st.button('Dictionary Audio Downloader'):
            session_state.page = "dictionary"
            st.experimental_rerun()

elif session_state.page == "task1":
    import task1
    if st.button("Back to main"):
        session_state.page = "main"
        st.experimental_rerun()

elif session_state.page == "dictionary":
    import get_dict
    if st.button("Back to main"):
        session_state.page = "main"
        st.experimental_rerun()
