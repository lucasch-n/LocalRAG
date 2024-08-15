from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def load_chunks(docs):
  chunks = []
  for doc in docs:
      raw_document = PyMuPDFLoader(doc).load()
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
      doc_chunks = text_splitter.split_documents(raw_document)
      chunks.extend([chunk for chunk in doc_chunks]) 
  return chunks

print("hello!")

embeddings = HuggingFaceEmbeddings(model_name="snowflake/snowflake-arctic-embed-m")

pdfdb = ['./DocLib/7.06__The_Quantum_Harmonic_Oscillator.pdf']
# collection.add(documents=load_chunks(pdfdb))
db = Chroma.from_documents(documents=load_chunks(pdfdb), embedding=embeddings, persist_directory='../chunkdbs')
print("PDFs indexed.")
db.persist()
