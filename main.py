# from llama_index.core import Settings
# from llama_index.core import PromptTemplate
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
# import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from openai import OpenAI
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import your RAG system components here
# from your_rag_module import retrieve_context
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

embeddings = HuggingFaceEmbeddings(model_name="snowflake/snowflake-arctic-embed-m")
vectordb = Chroma(persist_directory='../chunkdbs',embedding_function=embeddings)

def retrieve_context(query: str):
  docs = vectordb.similarity_search(query)
  search_results = ''.join([doc.page_content for doc in docs])

  response = client.chat.completions.create(
    model="llama3",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": query + search_results}
    ]
  )
  return response.choices[0].message.content
  # with st.chat_message("assistant"):
      # reply = st.write(response.choices[0].message.content)


# Load and chunk documents
def load_chunks(docs):
  


  chunks = []
  for doc in docs:
      raw_document = PyMuPDFLoader(doc).load()
      text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
      doc_chunks = text_splitter.split_documents(raw_document)
      for chunk in doc_chunks:
          chunks.append(chunk) 
  return chunks

# query = st.chat_input("Ask a question!")
# Load vector embedding model

app = FastAPI()

class Query(BaseModel):
    text: str

class Response(BaseModel):
    context: str

@app.post("/query", response_model=Response)
async def process_query(query: Query):
    try:
        # Call your RAG system here
        context = retrieve_context(query=query.text)
        return Response(context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




# r = requests.get('http://localhost:11434/v1/chat/completions', data={
#  "model": "llama3",
#  "messages": [
#    {"role":"system", "content":"You are an AI assistant that helps people find information."},
#    {"role":"user","content":"what is FRS?"}
#  ]
# })

# print(r.text)
# query = "What is a dual space?"
# text = "This is a test document."
# query_result = embeddings.embed_query(query)
# print(query_result[:3])

#{"model": "llama3", "prompt": [query].append(docs), "stream": False}