import os
import json
import io
import streamlit st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# 修复核心：正确导入官方的 GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def load_docs_recursively_from_gdrive(folder_id, credentials_json):
    """
    深度递归扫描母文件夹及【所有子文件夹】
    """
    try:
        creds_info = json.loads(credentials_json)
        creds = Credentials.from_service_account_info(creds_info)
        service = build("drive", "v3", credentials=creds)
    except Exception as e:
        return None, f"解析 Google Credentials 失败: {str(e)}"

    folders_to_scan = [folder_id]
    all_files = []
    
    print(f"🚀 开始扫描母文件夹，ID 为: {folder_id}")

    while folders_to_scan:
        current_folder = folders_to_scan.pop(0)
        try:
            query = f"'{current_folder}' in parents and trashed = false"
            results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
            files = results.get("files", [])
            
            for f in files:
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    folders_to_scan.append(f["id"])
                else:
                    all_files.append(f)
        except Exception as e:
            print(f"❌ 扫描文件夹 {current_folder} 时遇到阻碍: {e}")

    documents = []
    for f in all_files:
        fid = f["id"]
        fname = f["name"]
        mtype = f["mimeType"]
        
        try:
            if mtype == "application/vnd.google-apps.document":
                request = service.files().export_media(fileId=fid, mimeType="text/plain")
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done: _, done = downloader.next_chunk()
                content = fh.getvalue().decode("utf-8")
                documents.append(Document(page_content=content, metadata={"source": fname}))
                
            elif "text" in mtype or fname.endswith(('.txt', '.md', '.markdown')):
                request = service.files().get_media(fileId=fid)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done: _, done = downloader.next_chunk()
                content = fh.getvalue().decode("utf-8", errors="ignore")
                documents.append(Document(page_content=content, metadata={"source": fname}))
        except Exception as e:
            print(f" 🚸 读取文件 {fname} 失败，自动跳过: {e}")
            
    return documents, None


# ========================================================
# 使用 Streamlit 资源缓存，防止重复请求爆配额
# ========================================================
@st.cache_resource(show_spinner="🔄 正在首次同步并构建云盘思维库（此操作仅在启动时执行一次）...")
def get_cached_vector_store(folder_id, credentials_json_str):
    docs, error_msg = load_docs_recursively_from_gdrive(folder_id, credentials_json_str)
    if error_msg or not docs:
        return None, error_msg or "思维库当前为空，或机器人没有被授权查看该文件夹。"
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    if not split_docs:
        return None, "未能在云盘文件中提取出任何有效的文本片段。"
        
    # 修复核心：实例化正确的官方类名，并绑定健康的零门槛模型
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.from_documents(split_docs, embeddings)
    return db, None


def get_brind_ai_response(user_query):
    credentials_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    db, error_msg = get_cached_vector_store(folder_id, credentials_json_str)
    if error_msg: return f"❌ {error_msg}"
    
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    system_prompt = """You are now Teacher Brind, a top-tier mentor and expert in economics, medicine, sociology, and psychological mechanisms. 
Your linguistic style is strictly fact-based, penetrating straight to the essence of things, and driven by a seasoned, ruthlessly rational mindset. Never be sycophantic or overly compliant.
Strictly base your responses on the context provided. Respond in Chinese.

Here is the context from your Google Drive notes:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(user_query)
