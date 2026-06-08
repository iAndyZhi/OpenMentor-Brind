import os
import tempfile
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

def get_brind_ai_response(user_query):
    # Temporarily write the JSON credentials string from Streamlit Secrets to a file for Google Drive Loader
    credentials_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_cred_file:
        temp_cred_file.write(credentials_json_str)
        temp_cred_path = temp_cred_file.name

    try:
        # 1. Dynamically load the latest notes and chat logs from the Google Drive folder
        loader = GoogleDriveLoader(
            folder_id=os.environ.get("GOOGLE_DRIVE_FOLDER_ID"),
            service_account_key=temp_cred_path,
            recursive=False
        )
        docs = loader.load()
        
        # 2. Split high-density notes into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)
        
        # 3. Vectorize using Gemini's Embedding model and create an in-memory vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        db = FAISS.from_documents(split_docs, embeddings)
        
        # 4. Define the core persona Prompt for Teacher Brind
        system_prompt = """
        You are now Teacher Brind, a top-tier mentor and expert in economics, medicine, sociology, and psychological mechanisms. 
        
        Your linguistic style is strictly fact-based, penetrating straight to the essence of things, and driven by a seasoned, ruthlessly rational mindset. Never be sycophantic, overly compliant, or submissive to the user like a typical AI.
        
        You must strictly base your responses on your historical notes and chat logs retrieved from the knowledge base. If the user's viewpoint has logical fallacies, ruthlessly yet professionally point out their errors by breaking down the underlying biological, physical, economic, or sociological mechanisms. Do not use useless platitudes or polite nonsense.
        
        CRITICAL: Respond in Chinese, maintaining the exact persona described above.
        """
        
        # 5. Initialize the newly released Gemini 3.5 Flash for rapid & deep agentic reasoning
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=db.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": system_prompt}
        )
        return qa_chain.run(user_query)
    
    finally:
        # Clean up the temporary credentials file
        os.remove(temp_cred_path)
