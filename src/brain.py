import os
import json
import io
import google.generativeai as genai
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

class NativeGeminiEmbeddings:
    """使用原生 SDK，完美规避 LangChain 路由 Bug，并切换至最新兼容的 gemini-embedding-001 模型"""
    def __init__(self, model_name="models/gemini-embedding-001"):
        self.model_name = model_name
        api_key = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        
    def embed_documents(self, texts):
        if not texts: return []
        response = genai.embed_content(model=self.model_name, content=texts, task_type="retrieval_document")
        return response['embedding']
        
    def embed_query(self, text):
        response = genai.embed_content(model=self.model_name, content=text, task_type="retrieval_query")
        return response['embedding']


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

    # 使用队列实现自动全盘广度/深度递归遍历
    while folders_to_scan:
        current_folder = folders_to_scan.pop(0)
        try:
            query = f"'{current_folder}' in parents and trashed = false"
            results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
            files = results.get("files", [])
            
            print(f"📂 正在扫描目录层级，当前文件夹 ID: {current_folder}，发现 {len(files)} 个项目")
            
            for f in files:
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    print(f" └── 发现子文件夹: {f['name']} (ID: {f['id']})，已加入扫描队列")
                    folders_to_scan.append(f["id"])
                else:
                    print(f" └── 发现文件: {f['name']} (MimeType: {f['mimeType']})")
                    all_files.append(f)
        except Exception as e:
            print(f"❌ 扫描文件夹 {current_folder} 时遇到阻碍 (可能无权限): {e}")

    documents = []
    print(f"📦 扫描结束！共抓取到 {len(all_files)} 个待解析文件。开始读取内容...")
    
    for f in all_files:
        fid = f["id"]
        fname = f["name"]
        mtype = f["mimeType"]
        
        try:
            # 兼容 Google Docs 文档
            if mtype == "application/vnd.google-apps.document":
                request = service.files().export_media(fileId=fid, mimeType="text/plain")
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done: _, done = downloader.next_chunk()
                content = fh.getvalue().decode("utf-8")
                documents.append(Document(page_content=content, metadata={"source": fname}))
                print(f" ✅ 成功解析 Google Doc: {fname}")
                
            # 兼容手动上传的 md / txt 格式笔记
            elif "text" in mtype or fname.endswith(('.txt', '.md', '.markdown')):
                request = service.files().get_media(fileId=fid)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done: _, done = downloader.next_chunk()
                content = fh.getvalue().decode("utf-8", errors="ignore")
                documents.append(Document(page_content=content, metadata={"source": fname}))
                print(f" ✅ 成功解析本地笔记: {fname}")
        except Exception as e:
            print(f" 🚸 读取文件 {fname} 失败，自动跳过: {e}")
            
    return documents, None


def get_brind_ai_response(user_query):
    credentials_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    docs, error_msg = load_docs_recursively_from_gdrive(folder_id, credentials_json_str)
    if error_msg: return f"❌ {error_msg}"
    if not docs: return "❌ 思维库当前为空，或机器人没有被授权查看该 Google Drive 文件夹。请检查云盘共享设置！"
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    if not split_docs: return "❌ 未能在云盘文件中提取出任何有效的文本片段。"
    
    # 【核心修改点】：模型升级为最新的兼容版本
    embeddings = NativeGeminiEmbeddings(model_name="models/gemini-embedding-001")
    db = FAISS.from_documents(split_docs, embeddings)
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
