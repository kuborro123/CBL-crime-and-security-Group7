import time, pandas as pd
import streamlit as st

# 1️⃣ Page setup
st.set_page_config(page_title="Demo", page_icon="📊", layout="wide")

# 2️⃣ Sidebar controls
with st.sidebar:
    name = st.text_input("Your name", "Alice")
    show_chart = st.checkbox("Show chart", True)

# 3️⃣ Main content
st.title("Hello, " + name + " 👋")
st.write("This small app hits most of Streamlit’s core APIs.")

# 4️⃣ Cached data loader
@st.cache_data
def load():
    df = pd.DataFrame({"x": range(10), "y": [i**2 for i in range(10)]})
    time.sleep(1)             # simulate work
    return df

df = load()
st.dataframe(df)

# 5️⃣ Optional chart
if show_chart:
    st.line_chart(df.set_index("x"))

# 6️⃣ Chat section
st.header("Chat with the app")
with st.chat_message("assistant"):
    st.write("Ask me anything!")
user_msg = st.chat_input("Type your message here…")
if user_msg:
    with st.chat_message("user"):
        st.write(user_msg)

# 7️⃣ Footer
st.markdown("---\nMade with ❤️ using Streamlit")


