import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser

llm = ChatOpenAI(
    temperature=0.1,
)

has_transcript = os.path.exists("./.cache/video.txt")

@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4","mp3")
    command = [
        "ffmpeg",
        "-y", 
        "-i", 
        video_path, 
        "-vn",  
        audio_path,
    ]
    subprocess.run(command, check=True)

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="📝"
)

@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000

    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i+1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")

@st.cache_data()
def transcript_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file,"rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text_file.write(transcript["text"])

st.title("MeetingGPT")

st.markdown(
    """
Welcome to MeetingGPT, upload a vidoe and I will gvie you a transcript, a
summary and a chat bot to ask any question about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4","avi","mkv","mov"],)

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4","mp3")
        transcript_path = video_path.replace("mp4","txt")
        with open(video_path, "wb") as f:
            f.write(video.read())
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path,5,"./.cache/chunks")
        status.update(label="Transcribing audio...")
        transcript_chunks(chunks_folder,transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript","Summary","Q&A"])

    with transcript_tab:
        with open(transcript_path,"r") as file:
            st.write(file.read())
    
    with summary_tab:
        start = st.button("Generate summary")

        if start:

            loader = TextLoader(transcript_path)

            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=600,
                chunk_overlap=100,
            )
            docs = loader.load_and_split(text_splitter=splitter)
            
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:   
            """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke({
                "text": docs[0].page_content,
            })

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a
                certain point: {existing_summary}
                We have the opportunity to refine the existing
                summary (only if needed) with some more context
                below.
                --------------------------------
                {context}
                --------------------------------
                Given the new context, refine the original
                summary.
                If the context isn't useful, RETURN the
                original summary.

                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1}")
                    summary = refine_chain.invoke({
                        "existing_summary":summary,
                        "context":doc.page_content,
                    })
            st.write(summary)



