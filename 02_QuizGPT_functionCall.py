import streamlit as st
import json
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage


st.set_page_config(
    page_title="QuizGPT",
    page_icon="📝"
)

st.title("QuizGPT")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[function,]
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    merged_text = format_docs(_docs)

    messages = [
        SystemMessage(content="You are a helpful quiz generator."),
        HumanMessage(content=f"Create a quiz about {topic}. Each question must have **exactly 4 answer choices, Use the following context:\n\n{merged_text}")
    ]

    response = llm.invoke(messages)
    arguments_str = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(arguments_str)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(term)
    return docs



with st.sidebar:
    docs = None
    show_feedback = st.checkbox("Show feedback immediately", value=True)
    choice = st.selectbox("Choose what you want to use.", ("File", "Wikipedia Article"),)
    if choice == "File":
        file = st.file_uploader("Upload a .docx, .txt or .pdf file", type=["txt", "pdf", "docx"])
        if file:
            docs = split_file(file)
            topic = file.name
            #st.write(docs)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
                #st.write(docs)


if not docs:
    st.markdown(
        """
        Wlecome to Quiz GPT!

        I will make a quiz for you based on the file or Wikipedia article you provide.
        Please upload a file or enter a topic to search Wikipedia.

        Get started by selecting a file or entering a topic in the sidebar.
        """
    )

else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    
    #st.write(response)
    if show_feedback:
        for idx, question in enumerate(response["questions"]):
            st.write(f"**Q{idx+1}. {question['question']}**")

            selected = st.radio(
                "Select an option:",
                [a["answer"] for a in question["answers"]],
                key=f"q{idx}_choice",
                index=None
            )

            if selected and show_feedback:
                correct_answer = next((a for a in question["answers"] if a["correct"]), None)
                if {"answer": selected, "correct": True} in question["answers"]:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Wrong! Correct answer is: **{correct_answer['answer']}**")


    else:
        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio("Select an option.", [answer["answer"] for answer in question["answers"]], index=None)
                if {"answer": value, "correct":True} in question["answers"]:
                    st.success("Correct!")
                elif value is not None:
                    st.error("Wrong!")
            button = st.form_submit_button()
            
        




