import os
import tempfile
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# 💡 终极修正：不要去猜官方的拆包名字，直接从通用入口安全引入现代链
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain

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
        
        # 4. 构建严格的 Brind 老师人设
        system_prompt = """You are now Teacher Brind, a top-tier mentor and expert in economics, medicine, sociology, and psychological mechanisms. 
        
Your linguistic style is strictly fact-based, penetrating straight to the essence of things, and driven by a seasoned, ruthlessly rational mindset. Never be sycophantic, overly compliant, or submissive to the user like a typical AI.
        
You must strictly base your responses on your historical notes and chat logs retrieved from the knowledge base. If the user's viewpoint has logical fallacies, ruthlessly yet professionally point out their errors by breaking down the underlying biological, physical, economic, or sociological mechanisms. Do not use useless platitudes or polite nonsense.

CRITICAL: Respond in Chinese, maintaining the exact persona described above.

Here is the context from your Google Drive notes to help you answer:
{context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 5. 初始化高效的 Gemini 3.5 Flash 大模型
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
        
        # 6. 运行无视子版本演进的现代链组合
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 4}), question_answer_chain)
        
        # 运行链获取回答
        response = retrieval_chain.invoke({"input": user_query})
        return response["answer"]
        
    finally:
        # 无论成功与否，最后务必安全销毁存在云端服务器上的临时凭证文件
        if os.path.exists(temp_cred_path):
            try:
                os.remove(temp_cred_path)
            except Exception:
                pass
