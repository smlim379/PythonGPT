from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import streamlit as st
import numpy as np

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None
    
    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        #save_message(self.message, "ai")
        pass

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=0.1,
    streaming=False,
    callbacks= [ChatCallbackHandler()]
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

@st.cache_resource
def get_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@st.cache_resource
def get_cache():
    return {}

@st.cache_resource
def get_question_store():
    documents = [Document(page_content="example content", metadata={"source": "cache"})]
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

#def save_message(message, role):
#    if "role" is "human":
#        memory.chat_memory.add_user_message(message)
#    elif "role" is "ai":
#        memory.chat_memory.add_ai_message(message)

def find_similar_question(query, store, threshold=0.9):
    results = store.similarity_search(query, k=1)
    if not results:
        return None
    candidate = results[0].page_content
    score = np.dot(
        OpenAIEmbeddings().embed_query(query),
        OpenAIEmbeddings().embed_query(candidate)
    )
    return candidate if score >= threshold else None


def send_message(message,role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.write("save=ture") #save_message(message, role)

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


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )



if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        memory = get_memory()
        cache = get_cache()
        question_store = get_question_store()
        send_message("I'm ready! Ask away!.","ai", save=False)
        paint_history_from_memory(memory)

        query = st.chat_input("Ask a question to the website.")

        if query:
            
            with st.chat_message("human"):
                st.markdown(query)
            
            memory.chat_memory.add_user_message(query)
            

            # 1. Exact match
            if query in cache:
                answer = cache[query]
                if isinstance(answer, str):
                    send_message(answer, "ai")
                    memory.chat_memory.add_ai_message(answer)
                else:
                    st.error("Exact answer is not a string.") 
                st.stop()

            # 2. Similar match
            similar = find_similar_question(query, question_store)
            if similar and similar in cache:
                similar_answer = cache[similar]
                if isinstance(similar_answer, str):
                    send_message(f"(From similar question)\n{similar_answer}", "ai")
                    memory.chat_memory.add_ai_message(similar_answer)
                else:
                    st.error("Similar answer is not a string.")
                st.stop()

            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
                    
            with st.chat_message("ai"):
                response = chain.invoke(query)
                if isinstance(response.content, str):
                    answer = response.content
                    st.markdown(answer)
                    cache[query] = answer
                    question_store.add_texts(texts=[query])
                    memory.chat_memory.add_ai_message(answer)
                else:
                    st.error("Response content is not a string.")
                                     
               
            
 
 

            #siteuse: https://htmlburger.com/sitemap.xml
