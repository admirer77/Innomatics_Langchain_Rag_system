from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


def rag(user_input):
    f = open(r'M:\innomatics\Gemini_API_Key.txt')

    GOOGLE_API_KEY = f.read()

    # Set the OpenAI Key and initialize a LLM model
    llm = GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-pro-latest", temperature=1)


    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(r"M:\innomatics\project6\2404.07143.pdf")
    pages = loader.load_and_split()

    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

    db_conn = Chroma(persist_directory=r"M:\innomatics\project6\chroma_db_", embedding_function=embeddings)

    retriever = db_conn.as_retriever(search_kwargs={"k":5})



    output_parser = StrOutputParser()

    chat_template = ChatPromptTemplate.from_messages([
        # System Message Prompt Templatea
        SystemMessage(content="""You are a Helpful AI Bot. 
        You take the context and question from user. Your answer should be based on the specific context."""),
        # Human Message Prompt Template
        HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
        Context:
        {context}
        
        Question: 
        {question}
        
        Answer: """)
    ])

    from langchain_core.runnables import RunnablePassthrough

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | llm
        | output_parser
    )

    response = rag_chain.invoke(user_input)

    return response