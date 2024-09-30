# 2:02:53
import os
import streamlit as st
import pickle
import time
import langchain
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title("New Research Tool.. &#9917; &#9924; &#9731;")

st.sidebar.title("News article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if process_url_clicked:
    if urls:

        # load data
        main_placeholder.text("Data Loading.. Started.. &#10003; &#10004; &#9989;")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        #split data
        main_placeholder.text("Text Splitter.. Started.. &#10003; &#10004; &#9989;")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=500,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(data)

        # create embeddings and vectorstore
        embeddings = OpenAIEmbeddings()
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector.. Started.. &#10003; &#10004; &#9989;")

        # save the vectorstore
        time.sleep(2)
        # save the faiss index
        vectorindex_openai.save_local("vectorstore")
        main_placeholder.text("Vector Store.. Saved.. &#10003; &#10004; &#9989;")


query = st.text_input("Question: ")
if query:
    llm = OpenAI(temperature=0.9, max_tokens=500)
    x = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = x.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    main_placeholder.text("Answering.. Started.. &#10003; &#10004; &#9989;")
    result = chain({"question": query}, return_only_outputs=True)
    print(result)
    st.header("Answer:")
    st.write(result["answer"])
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")
        for source in sources_list:
            st.write(source)
    else:
        st.write("Please process URLs first.")


