import os
import tempfile
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def get_brind_ai_response(user_query):
    # 从 Streamlit Secrets 读取 Google 服务账号 JSON 字符串
    credentials_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    
    # 使用 NamedTemporaryFile 创建临时凭证文件，供 GoogleDriveLoader 读取
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_cred_file:
        temp_cred_file.write(credentials_json_str)
        temp_cred_path = temp_cred_file.name

    try:
        # 1. 动态从指定的 Google Drive 文件夹加载最新的老师笔记/聊天记录
        loader = GoogleDriveLoader(
            folder_id=os.environ.get("GOOGLE_DRIVE_FOLDER_ID"),
            service_account_key=temp_cred_path,
            recursive=False
        )
        docs = loader.load()
        
        # 2. 将高密度的笔记文本切分成适合 AI 检索的小片段（Chunks）
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)
        
        # 3. 使用 Gemini 官方的 Embedding 模型进行向量化，并在内存中建立本地向量库
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        db = FAISS.from_documents(split_docs, embeddings)
        
        # 4. 精心打造 Brind 老师的核心人设 PromptTemplate
        # 注意：这里必须显式包含 {context}（知识库内容）和 {question}（用户提问）占位符
        template = """You are now Teacher Brind, a top-tier mentor and expert in economics, medicine, sociology, and psychological mechanisms. 
        
Your linguistic style is strictly fact-based, penetrating straight to the essence of things, and driven by a seasoned, ruthlessly rational mindset. Never be sycophantic, overly compliant, or submissive to the user like a typical AI.
        
You must strictly base your responses on your historical notes and chat logs retrieved from the knowledge base. If the user's viewpoint has logical fallacies, ruthlessly yet professionally point out their errors by breaking down the underlying biological, physical, economic, or sociological mechanisms. Do not use useless platitudes or polite nonsense.

CRITICAL: Respond in Chinese, maintaining the exact persona described above.

Context from Google Drive:
{context}

Question: {question}
Answer:"""

        brind_prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )
        
        # 5. 初始化高效的 Gemini 3.5 Flash 大模型
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
        
        # 6. 构建带有知识检索和严格人设的问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=db.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": brind_prompt}
        )
        
        # 运行问答链获取 Brind 老师的回答
        return qa_chain.run(user_query)
        
    finally:
        # 无论成功与否，最后务必安全销毁存在云端服务器上的临时凭证文件，确保隐私与安全
        if os.path.exists(temp_cred_path):
            try:
                os.remove(temp_cred_path)
            except Exception:
                pass
