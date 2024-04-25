# Because of some chromadb installation issue in Jupyter Notebook, this file is created.


from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma

f = open(r'M:\innomatics\Gemini_API_Key.txt')

GOOGLE_API_KEY = f.read()

# Set the OpenAI Key and initialize a LLM model
llm = GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-pro-latest", temperature=1)



loader = PyPDFLoader(r"M:\innomatics\project6\2404.07143.pdf")
pages = loader.load_and_split()


text_splitter = NLTKTextSplitter(chunk_size=300)
texts = text_splitter.split_documents(pages)

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")



# Embed each chunk and load it into the vector store
db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db_")

# Persist the database on drive
db.persist()