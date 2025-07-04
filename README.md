Here RAG means Retrivel Agumented Generation........

Here, RAG is A Wonderful Application To Create a Question Retrivers And Find the Output of that....And Iam doing the RAG application Google Colab...

Step 1 : Install the Packages and Import the Packages What you Want to create a RAG Applictaion

        Here I am taking Chormadb Data Base 


        # installing Packages

        !pip install langchain                    # chains,memory
        !pip install chromadb                     # save vector data base
        !pip install langchain-community          # tools,memory,chunks,load
        !pip install langchain-google-genai       # by langchain installing google genai models


        # importing packages

        from langchain.prompts import PromptTemplate                         # prompt template
        from langchain.vectorstores import Chroma                            # save the vector data base
        from langchain.text_splitter import RecursiveCharacterTextSplitter   # Chunk the Data
        from langchain.document_loaders import TextLoader                    # Load the Data like, Pdf,txt,word any documents
        from langchain.chains import VectorDBQA,LLMChain,RetrievalQA         # vectorDBQA - vector data base, RetrieverQA - Asking Question and converting multiple questions , LLMChain -  for classification
        from langchain.retrievers.multi_query import MultiQueryRetriever     # Multiple Answers by 
        from langchain_google_genai import ChatGoogleGenerativeAI            # GenAI Model to Retrive
        from langchain_google_genai import GoogleGenerativeAIEmbeddings      # Embedding models


Step 2 : Load the Data (pdf,csv,txt,etc...)

        # Load documents
        file_path=r'/content/State_union.txt'
        loader=TextLoader(file_path,encoding='utf-8-sig')
        documents=loader.load()
        len(documents)
        
Step 3 : Chunk the Data (Splitting the Data )

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        len(texts)

Step 4 : set up Embbedings

         
        embeddings=GoogleGenerativeAIEmbeddings(
            model='models/embedding-001',
            google_api_key='',
            task_type='retrieval_query',
        )

Step 5 : Create the Vector Store

        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)
        vectordb

Step 6 : Make Prompt and prompt Template 

Step 7 : Create the QA chains (Questions)

Step 8 : Run the Query.......

        
     
