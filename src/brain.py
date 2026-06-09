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

# 💡 核心武器：手写一个原生谷歌向量类，完美平替 LangChain 报错组件
class NativeGeminiEmbeddings:
    """
    使用 Google 官方原生 SDK 的向量生成器，
    完美避开 LangChain 底层硬编码 v1beta 导致的 404 恶性 Bug。
    """
    def __init__(self, model_name="models/text-embedding-004"):
        self.model_name = model_name
        # 确保配置了你的 Gemini API Key
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        
    def embed_documents(self, texts):
        if not texts:
            return []
        response = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type="retrieval_document"
        )
        return response['embedding']
        
    def embed_query(self, text):
        response = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']


def load_docs_recursively_from_gdrive(folder_id, credentials_json):
    """
    官方原生 Google Drive 读取器：深度扫描所有子文件夹
    """
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
            results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
            files = results.get("files", [])
            for f in files:
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    folders_to_scan.append(f["id"])
                else:
                    all_files.append(f)
        except Exception as e:
            print(f"扫描子文件夹 {current_folder} 时跳过 (原因: {e})")

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
                while not done:
                    _, done = downloader.next_chunk()
                content = fh.getvalue().decode("utf-8")
                documents.append(Document(page_content=content, metadata={"source": fname}))
                
            elif "text" in mtype or fname.endswith(('.txt', '.md', '.markdown')):
                request = service.files().get_media(fileId=fid)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                content = fh.getvalue().decode("utf-8", errors="ignore")
                documents.append(Document(page_content=content, metadata={"source": fname}))
        except Exception as e:
            print(f"读取文件 {fname} 失败，已自动跳过 (原因: {e})")
            
    return documents, None


def get_brind_ai_response(user_query):
    credentials_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    # 1. 启动全盘深度扫描
    docs, error_msg = load_docs_recursively_from_gdrive(folder_id, credentials_json_str)
    if error_msg:
        return f"❌ {error_msg}"
        
    if not docs:
        return "❌ Brind 老师的思维库当前为空，或者服务账号没有被授权查看该 Google Drive 文件夹。"
        
    # 2. 切分笔记
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    
    if not split_docs:
        return "❌ 未能在云盘文件中提取出任何有效的字符片段。"
    
    # 3. 💡 核心改动：使用我们手写的原生安全向量引擎构建 FAISS 数据库
    embeddings = NativeGeminiEmbeddings(model_name="models/text-embedding-004")
    db = FAISS.from_documents(split_docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    # 4. 完美注入硬核不谄媚的 Teacher Brind 导师人设
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
    
    # 5. 呼叫 Gemini 3.5 Flash 引擎
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    
    # 6. 工作流闭环
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(user_query)
