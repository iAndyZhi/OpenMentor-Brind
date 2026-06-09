import os
import json
import io
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# 保持官方正确包装器导入
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def load_docs_recursively_from_gdrive(folder_id, credentials_json):
    """
    深度递归扫描母文件夹及【所有子文件夹】
    """
    # 【防御性代码】：如果你不小心把整个云盘 URL 贴进了 Secrets，这里会自动切出干净的纯 ID
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
    
    print(f"🚀 开始扫描母文件夹，ID 为: {folder_id}")

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
            # 【诊断升级】：不再静默吞掉异常，如果是 API 报错（例如 404/403）直接顶到前端显示
            return None, f"谷歌云盘接口请求失败。当前扫描的文件夹 ID 为: {current_folder}。错误明细: {str(e)}"

    if not all_files:
        return None, f"机器人成功进入了云盘，但在该文件夹(ID: {folder_id})内没有找到任何文件。请确认文件夹内是否有内容。"

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
            
    # 【诊断升级】：如果是由于文件格式不匹配导致列表为空，给出极其精准的诊断提示
    if not documents and skipped_files:
        return None, f"⚠️ 在云盘中找到了 {len(skipped_files)} 个文件，但它们的格式目前无法被机器人解析。目前系统仅支持直接提取『Google 文档(Google Docs)』或『.txt/.md 纯文本』。请在云盘中将非标文件右键『另存为 Google 文档』后再试。\n\n检测到的文件列表: {', '.join(skipped_files)}"
        
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
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.from_documents(split_docs, embeddings)
    return db, None


def get_brind_ai_response(user_query):
    # 【核心改进】：全面从 os.environ 切换到 Streamlit 标准的 st.secrets，确保本地和云端读取绝对稳定
    credentials_json_str = st.secrets.get("GOOGLE_CREDENTIALS_JSON")
    folder_id = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
    
    if not credentials_json_str:
        return "❌ 部署错误：未在 Streamlit Secrets 中成功读取到 GOOGLE_CREDENTIALS_JSON"
    if not folder_id:
        return "❌ 部署错误：未在 Streamlit Secrets 中成功读取到 GOOGLE_DRIVE_FOLDER_ID"
    
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
