# basic-gen-ai-example

1. Project is expected to take 2 (or) 3 URLs as inputs.
2. Process the input URLs. Via UnstructuredURLLoader
3. Input will be converted to chunks. Via RecursiveCharacterTextSplitter > Create docs
4. "docs" i.e. chunks will be then be processed to for Vector Store. Via FAISS.from_documents(docs, OpenAIEmbeddings())
5. Vector Stores will be saved locally 
6. "chain" will be created via RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = retriever). Refer below:
    x = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = x.as_retriever()
7. "chain" will then be used for asking question. Refer below:
    query = "whats is the market cap of Nvidia?"
    langchain.globals.set_debug(True)
    chain({"question": query}, return_only_outputs=True)

# steps to run
1. install dependencies using below (ONE TIME)
   pip install -r requirement.txt
2. Run project using below command
    streamlit run main.py

# Personal Github: https://github.com/ved740/GenAI-QuestionAnsweringBasedOnURLs

# Input URLs
1. https://finance.yahoo.com/news/indias-tata-motors-slumps-ubs-062715228.html
2. https://www.livemint.com/market/stock-market-news/tata-motors-vs-maruti-vs-m-m-which-auto-stock-to-buy-amid-rbis-rate-cut-buzz-11727509659377.html
3. https://www.livemint.com/market/stock-market-news/stocks-to-buy-or-sell-dharmesh-shah-of-icici-securities-recommends-tata-motors-and-deepak-nitrite-to-buy-on-september-11727539077898.html