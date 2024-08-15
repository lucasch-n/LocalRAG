import streamlit as st
import requests
import json
query = st.chat_input("Ask a question!")

url = "http://localhost:8000/query"
headers = {
    "Content-Type": "application/json"
}
data = {
    "text": query
}



if query is not None:
    response = requests.post(url, json=data, headers=headers)
    with st.chat_message("assistant"):
        reply = st.write(json.loads(response.content)['context'])

