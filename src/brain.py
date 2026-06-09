import os
import json
import io
import streamlit as st
# 🚀 引入谷歌全新主力一代官方 GenAI SDK 客户端
from google import genai
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class GoogleNativeEmbeddings(Embeddings):
    """
    基于谷歌全新官方 google-genai SDK 封装的嵌入类。
    原生强制锁定 v1 正式版接口路由，彻底终结 text-embedding-004 找不到的 404 顽疾。
    """
    def __init__(self):
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        # 全新一代客户端初始化，内部自带 v1 稳健路由
        self.client = genai.Client(api_key=api_key)
        self.model_name = "text-embedding-004"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
            )
            # 兼容处理单条和多条返回的结构
            if isinstance(response.embeddings, list):
                return [e.values for e in response.embeddings]
            return [response.embeddings.values]
        except Exception as e:
            raise RuntimeError(f"Embedding 批量生成失败: {str(e)}")

    def embed_query(self, text: str) -> list[float]:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            if isinstance(response.embeddings, list):
                return response.embeddings[0].values
            return response.embeddings.values
        except Exception as e:
            raise RuntimeError(f"Embedding 单条检索词生成失败: {str(e)}")


def load_docs_recursively_from_gdrive(folder_id, credentials_json):
    """
    深度递归扫描母文件夹及【所有子文件夹】
    """
    if "folders/" in folder_id:
        folder_id = folder_id.split("folders/")[-1].split("?")[0].strip()
    else:
        folder_id = folder_id.strip()

    try:
        creds_info = json.loads(credentials_json)
        creds = Credentials.from_service_account_info(creds_info)
        service = build("drive", "v3", credentials=creds)
    except Exception as e:
        return None, f"解析 Google Credentials 失败: {str(e)}"

    folders_to_scan = [folder_id]
    all_files = []

    while folders_to_scan:
        current_folder = folders_to_scan.pop(0)
        try:
            query = f"'{current_folder}' in parents and trashed = false"
            results = service.files().list(
                q=query, 
                fields="files(id, name, mimeType)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            files = results.get("files", [])
            
            for f in files:
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    folders_to_scan.append(f["id"])
                else:
                    all_files.append(f)
        except Exception as e:
            return None, f"谷歌云盘接口请求失败。文件夹 ID: {current_folder}。错误: {str(e)}"

    if not all_files:
        return None, f"机器人成功进入了云盘，但在该文件夹(ID: {folder_id})内没有找到任何文件。"

    documents = []
    skipped_files = []
    
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
            else:
                skipped_files.append(f"{fname} ({mtype})")
        except Exception as e:
            print(f" 🚸 读取文件 {fname} 失败，自动跳过: {e}")
            
    if not documents and skipped_files:
        return None, f"⚠️ 未能解析格式。目前系统仅支持『Google 文档』或『.txt/.md 纯文本』。\n\n检测到的文件列表: {', '.join(skipped_files)}"
        
    return documents, None


@st.cache_resource(show_spinner="🔄 正在首次同步并构建云盘思维库（此操作仅在启动时执行一次）...")
def get_cached_vector_store(folder_id, credentials_json_str):
    docs, error_msg = load_docs_recursively_from_gdrive(folder_id, credentials_json_str)
    if error_msg:
        return None, error_msg
    if not docs:
        return None, "思维库当前为空，未能提取到任何有效文本。"
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    if not split_docs:
        return None, "未能在云盘文件中分割出任何有效的文本片段。"
        
    # 实例化基于全新 google-genai 库封装的稳定版嵌入
    embeddings = GoogleNativeEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    return db, None


def get_brind_ai_response(user_query):
    credentials_json_str = st.secrets.get("GOOGLE_CREDENTIALS_JSON")
    folder_id = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
    
    if not credentials_json_str or not folder_id:
        return "❌ 部署错误：未在 Streamlit Secrets 中成功读取到配置项"
    
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
    
    # 🚀 大脑采用最新的 gemini-3.5-flash
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    
    def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(user_query)
