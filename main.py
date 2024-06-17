#%%
# Imports 

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os 
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_community.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class Chatbot(): 
  # Contants 
  update_index = False # make it false when not wanting to update the index values

  load_dotenv() # load env variables

  # Load the boook 
  loader = TextLoader('./The Prince by Nicolo Machiavelli.txt') 
  documents = loader.load()

  # Split text 
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " "])
  # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
  docs = text_splitter.split_documents(documents)

  # Get embeddings from hugging face
  embeddings = HuggingFaceEmbeddings()



  # Initialize Pinecone client
  pinecone.init(
      api_key= os.getenv('PINECONE_API_KEY'),
      environment='gcp-starter'
  )

  # Define Index Name
  index_name = "langchain-demo"

  # Delete the index if it exists 
  if index_name in pinecone.list_indexes() and update_index is True:
      pinecone.delete_index(index_name)

  # Checking Index
  if index_name not in pinecone.list_indexes():
    # Create new Index
    pinecone.create_index(name=index_name, metric="cosine", dimension=768)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
  else:
    # Link to the existing index
    docsearch = Pinecone.from_existing_index(index_name, embeddings)


  # Define the repo ID and connect to Mixtral model on Huggingface
  repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
  llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.8, "top_k": 50}, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
  )

  template = """
  You are Nicolo Machiavelli. These Human will ask you a questions about your book 'The Prince'. 
  Use following piece of context to answer the question. 
  If you don't know the answer, just say you don't know. 
  Keep the answer concise.

  Context: {context}
  Question: {question}
  Answer: 

  """

  prompt = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
  )

  rag_chain = (
    {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
  )

bot = Chatbot()
input = input("Any me anything about the book 'The Prince by Nicolo Machiavelli': ")
result = bot.rag_chain.invoke(input)
print(result)