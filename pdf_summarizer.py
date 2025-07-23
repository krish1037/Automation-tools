import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

#  here we are setting the environment for the googl api key 
os.environ["GOOGLE_API_KEY"] = "Enter your own API key"

# load PDF
pdf_path = "Enter your own path of the folder "
loader = PyPDFLoader(pdf_path) # loading the pdf file using the PyPDFLoader from langchain_community
documents = loader.load()

# split text for better processing LLMs can't handle huge amont of text or strings at a time 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# defining the llm model as gemini-1.5-flash
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# QA chain for the connectivity purpose with the  ajent 
chain = load_qa_chain(llm, chain_type="stuff")

# giving the desired resut what user want 
query = "Give me a summary of the pdf in bullet points."
response = chain.run(input_documents=docs, question=query)

print("Summary:\n", response)
