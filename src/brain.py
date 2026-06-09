import os
import tempfile
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_brind_ai_response(user_query):
    # Fetch Google Credentials JSON string from Streamlit Secrets
    credentials_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    
    # Create a temporary file to hold credentials for GoogleDriveLoader
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_cred_file:
        temp_cred_file.write(credentials_json_str)
        temp_cred_path = temp_cred_file.name

    try:
        # 1. Dynamically load documents from the specified Google Drive folder
        loader = GoogleDriveLoader(
            folder_id=os.environ.get("GOOGLE_DRIVE_FOLDER_ID"),
            service_account_key=temp_cred_path,
            recursive=False
        )
        docs = loader.load()

        # SAFEGUARD: Check if the folder is empty or inaccessible
        if not docs:
            return "❌ 思维库当前为空，或account没有被授权查看该 Google Drive folder。请检查云盘共享设置！"
        
        # 2. Split high-density documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        if not split_docs:
            return "❌ 未能在文件夹中解析出任何有效文本片段，请确保文件内含有可读的文本内容。"
        
        # 3. Vectorize text chunks and initialize FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        db = FAISS.from_documents(split_docs, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 4})
        
        # 4. Construct Brind's rigorous persona
        system_prompt = """You are now Teacher Brind, a top-tier mentor and expert in economics, medicine, sociology, and psychological mechanisms. 
        
Your linguistic style is strictly fact-based, penetrating straight to the essence of things, and driven by a seasoned, ruthlessly rational mindset. Never be sycophantic, overly compliant, or submissive to the user like a typical AI.
        
You must strictly base your responses on your historical notes and chat logs retrieved from the knowledge base. If the user's viewpoint has logical fallacies, ruthlessly yet professionally point out their errors by breaking down the underlying biological, physical, economic, or sociological mechanisms. Do not use useless platitudes or polite nonsense.

CRITICAL: Respond in Chinese, maintaining the exact persona described above.

Here is the context from your Google Drive notes to help you answer:
{context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        # 5. Gemini 3.5 Flash 
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
        
        # 6. Assemble the RAG pipeline using LCEL syntax
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Execute chain and return the final response
        return rag_chain.invoke(user_query)
        
    finally:
        # Securely destroy the temporary credential file in all circumstances
        if os.path.exists(temp_cred_path):
            try:
                os.remove(temp_cred_path)
            except Exception:
                pass
