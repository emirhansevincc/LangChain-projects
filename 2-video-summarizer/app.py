import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA


os.environ['OPENAI_API_KEY'] = 'YourAPIKey'

llm = OpenAI()

def main():
    st.set_page_config(page_title='Project-2')

    st.header('Video Summarizer')

    video_url = st.text_input(
        ' This application helps users understand the content of YouTube videos by providing accurate summaries using the OpenAI API.', placeholder='Enter your URL here'
    )

    question = st.text_input(
        'What is your question?',
        placeholder='Enter your question here',
    )

    

    if video_url is not None and question != '':
        loader = YoutubeLoader.from_youtube_url(
            f'{video_url}', 
            add_video_info=True
        )

        result = loader.load()
        
        video_title = result[0].metadata['title']
        video_description = result[0].metadata['description']
        video_author = result[0].metadata['author']
        video_publish_date = result[0].metadata['publish_date']
        video_length = result[0].metadata['length']
        video_view_count = result[0].metadata['view_count']

        # Calculate the video format
        hours = video_length // 3600
        minutes = (video_length % 3600) // 60
        seconds = (video_length % 3600) % 60

        st.write(f"""
            Information about the video
            - Title: {video_title}
            - Description: {video_description}
            - Author: {video_author}
            - Publish Date: {video_publish_date}
            - Length: {hours} hours, {minutes} minutes, {seconds} seconds
            - View Count: {video_view_count}
        """)

        result = loader.load()

        # We create a text splitter object that will split the text into chunks of 2000 characters. It's for long texts.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=0
        )

        texts = text_splitter.split_documents(result)

        embeddings = OpenAIEmbeddings()

        # vector_store object is a tool for smilarity-based searching among your embedded texts
        vector_store = FAISS.from_documents(texts, embeddings)

        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="map_reduce", 
            retriever=vector_store.as_retriever()
        )

        if question != '':
            st.write(qa.run(f'{question}'))

main()