from langchain_openai import ChatOpenAI
#from langchain.document_loaders import UnstructuredLoader
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langsmith import traceable
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import AIMessage, HumanMessage
import time
import streamlit as st

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
    

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks= [ChatCallbackHandler()],
                 )

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📝",
)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)
    
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver

def save_message(message,role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message,role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history_from_memory(memory):
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("human"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("ai"):
                st.markdown(msg.content)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

@st.cache_resource
def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

prompt = ChatPromptTemplate.from_messages([
    ("system","""Answer the question using ONLY the following context. If you don't know the answer
     just say you don't know. DON'T make anything up.
     
     Context: {context}""",),
     ("human","{question}"),
])

st.title("DocumentGPT")

st.markdown("""
Welcome!
            
Use this chatbot to ask question to an AI about your files!
            
Upload your files on the sidebar.
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file",type=["pdf","txt","docx"])

if file:
    retriever = embed_file(file)

    memory = get_memory()

    send_message("I'm ready! Ask away!","ai", save=False)

    chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
        )

    paint_history_from_memory(memory)

    message = st.chat_input("Ask anything about your file...")

    

    if message:
        with st.chat_message("human"):
            st.markdown(message)
        
        with st.chat_message("ai"):
            response = chain.invoke({"question":message})
            st.markdown(response["answer"])
        
        #send_message(response.content, "ai")

    
    
