import os
import io
import json
import shutil
import pdfplumber
import docx
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from uuid import uuid4

# LangChain and FAISS imports
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5174", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables and FAISS Setup ---
# Create an absolute path to the FAISS index, right next to our script
SCRIPT_DIR = Path(__file__).resolve().parent
FAISS_INDEX_PATH = str(SCRIPT_DIR / "faiss_index")

# --- Persistent File Storage Setup ---
UPLOADED_TEXTS_DIR = SCRIPT_DIR / "uploaded_texts"
FILE_MAP_PATH = SCRIPT_DIR / "file_map.json"

# Ensure directories exist
UPLOADED_TEXTS_DIR.mkdir(parents=True, exist_ok=True)

def load_file_map():
    """Loads the file mapping from disk."""
    if FILE_MAP_PATH.exists():
        with open(FILE_MAP_PATH, "r") as f:
            return json.load(f)
    return {}

def save_file_map(file_map):
    """Saves the file mapping to disk."""
    with open(FILE_MAP_PATH, "w") as f:
        json.dump(file_map, f, indent=4)

# Load file map on startup
file_map_data = load_file_map()

# Initialize clients
embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
llm = ChatOllama(model="llama3:8b")

# Summarization chain
summarization_prompt_template = """
Summarize the following text in a few sentences:
{text}
"""
summarization_prompt = ChatPromptTemplate.from_template(summarization_prompt_template)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

# Query transformation chain
query_transform_prompt_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Question: {question}
Standalone question:
"""
query_transform_prompt = ChatPromptTemplate.from_template(query_transform_prompt_template)
query_transform_chain = LLMChain(llm=llm, prompt=query_transform_prompt)

# In-memory store for conversation histories
conversation_histories = {}

# Global variable to hold the vector store
vectorstore: Optional[FAISS] = None

@app.on_event("startup")
def load_faiss_index():
    """Load the FAISS index from disk on application startup if it exists."""
    global vectorstore
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("No existing FAISS index found. A new one will be created on first upload.")

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the DocuMind API!"}

@app.post("/clear")
def clear_knowledge_base():
    """Clears the existing FAISS vector store from memory and disk."""
    global vectorstore, file_map_data
    vectorstore = None
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
            print(f"Successfully deleted FAISS index directory: {FAISS_INDEX_PATH}")
        
        # Clear uploaded texts and file map
        if UPLOADED_TEXTS_DIR.exists():
            shutil.rmtree(UPLOADED_TEXTS_DIR)
            UPLOADED_TEXTS_DIR.mkdir(parents=True, exist_ok=True) # Recreate empty directory
            print(f"Successfully deleted uploaded texts directory: {UPLOADED_TEXTS_DIR}")
        file_map_data.clear()
        save_file_map(file_map_data)
        print("File map cleared.")

        return {"message": "Knowledge base and uploaded files cleared successfully."}
    except Exception as e:
        print(f"Error clearing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing knowledge base: {e}")

@app.get("/list_files")
def list_files_in_knowledge_base():
    """Lists the unique filenames currently stored in the knowledge base."""
    global file_map_data
    files_info = []
    for file_id, file_info in file_map_data.items():
        files_info.append({"id": file_id, "filename": file_info["original_filename"]})
    return {"files": files_info}

@app.delete("/delete_file/{file_id}")
def delete_file_from_knowledge_base(file_id: str):
    """Deletes a specific file and its associated chunks from the knowledge base."""
    global vectorstore, file_map_data

    if file_id not in file_map_data:
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found.")

    file_info = file_map_data[file_id]
    original_filename = file_info["original_filename"]
    text_file_path = Path(file_info["text_file_path"])

    try:
        # 1. Delete the raw text file
        if text_file_path.exists():
            os.remove(text_file_path)
            print(f"Deleted text file: {text_file_path}")
        
        # 2. Remove from file map
        del file_map_data[file_id]
        save_file_map(file_map_data)
        print(f"Removed {original_filename} from file map.")

        # 3. Rebuild FAISS index from remaining files
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
            print(f"Cleared old FAISS index for rebuild: {FAISS_INDEX_PATH}")
        
        all_current_docs_for_rebuild = []
        for f_id, f_info in file_map_data.items():
            current_text_path = Path(f_info["text_file_path"])
            if current_text_path.exists():
                with open(current_text_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.create_documents([full_text])
                for doc in docs:
                    doc.metadata["source"] = f_info["original_filename"]
                    doc.metadata["file_id"] = f_id
                    if f_info["original_filename"].endswith('.pdf') and 'page_number' in doc.metadata:
                        doc.metadata["page"] = doc.metadata["page_number"]
                all_current_docs_for_rebuild.extend(docs)
            else:
                print(f"Warning: Text file not found for {f_info['original_filename']} ({f_id}). Skipping during rebuild.")

        if not all_current_docs_for_rebuild:
            vectorstore = None # No documents left, clear vectorstore
            print("No documents remaining after rebuild, FAISS index cleared.")
        else:
            vectorstore = FAISS.from_documents(all_current_docs_for_rebuild, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)
            print(f"FAISS index rebuilt and saved to {FAISS_INDEX_PATH} with {len(all_current_docs_for_rebuild)} chunks.")

        return {"message": f"File '{original_filename}' deleted and knowledge base updated successfully."}

    except Exception as e:
        print(f"Error deleting file {original_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    print(f"Received {len(files)} files for upload.") # New print statement
    global vectorstore, file_map_data # Access global file_map_data
    all_docs = []
    full_text_for_summary = ""
    
    newly_uploaded_files_info = [] # To store info about files just uploaded

    for file in files:
        file_id = str(uuid4()) # Generate a unique ID for this file
        original_filename = file.filename
        text_file_path = UPLOADED_TEXTS_DIR / f"{file_id}.txt"

        file_bytes = await file.read()
        try:
            # Determine file type and extract text
            if original_filename.endswith('.pdf'):
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    full_text = "".join(page.extract_text() or "" for page in pdf.pages)
            elif original_filename.endswith('.txt'):
                full_text = file_bytes.decode('utf-8')
            elif original_filename.endswith('.docx'):
                try:
                    import docx # Import docx here to avoid global import if not used
                except ImportError:
                    raise HTTPException(status_code=500, detail="'python-docx' is not installed. Please install it with 'pip install python-docx'")
                doc = docx.Document(io.BytesIO(file_bytes))
                full_text = "\n".join([para.text for para in doc.paragraphs])
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {original_filename}. Only .pdf, .txt, and .docx are supported.")

            if not full_text.strip():
                raise HTTPException(status_code=400, detail=f"Could not extract text from {original_filename}. The file might be empty or image-based.")
            
            # Save the extracted text to disk
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            # Update file map
            file_map_data[file_id] = {
                "original_filename": original_filename,
                "text_file_path": str(text_file_path),
                "chunks_indexed": 0 # Placeholder, actual chunk count will be known after splitting
            }
            newly_uploaded_files_info.append({"id": file_id, "filename": original_filename})

            full_text_for_summary += full_text + "\n\n"
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.create_documents([full_text])
            
            # Add metadata to documents for source tracking
            for doc in docs:
                doc.metadata["source"] = original_filename
                doc.metadata["file_id"] = file_id # Link chunk to its file_id
                if original_filename.endswith('.pdf') and 'page_content' in doc.metadata: # Check for page_content in metadata
                    # pdfplumber adds 'page_number' to metadata, not 'page_content'
                    if 'page_number' in doc.metadata:
                        doc.metadata["page"] = doc.metadata["page_number"]
                # For TXT/DOCX, we don't have inherent page numbers per chunk, so we omit it.
            all_docs.extend(docs)
            file_map_data[file_id]["chunks_indexed"] = len(docs) # Update chunk count

        except Exception as e:
            # Clean up partially saved text file if error occurs
            if text_file_path.exists():
                os.remove(text_file_path)
            if file_id in file_map_data:
                del file_map_data[file_id]
            raise HTTPException(status_code=500, detail=f"Failed to process {original_filename}: {e}")

    if not all_docs:
        raise HTTPException(status_code=400, detail="No processable text found in the uploaded file(s).")

    # Rebuild FAISS index from all currently tracked files
    # This is crucial for deletion to work by re-indexing only existing files
    try:
        # Clear existing FAISS index on disk before rebuilding
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
            print(f"Cleared old FAISS index for rebuild: {FAISS_INDEX_PATH}")
        
        # Collect all text from currently tracked files to rebuild index
        all_current_docs_for_rebuild = []
        for f_id, f_info in file_map_data.items():
            text_path = Path(f_info["text_file_path"])
            if text_path.exists():
                with open(text_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.create_documents([full_text])
                for doc in docs:
                    doc.metadata["source"] = f_info["original_filename"]
                    doc.metadata["file_id"] = f_id
                    # Re-add page numbers for PDFs if available in original metadata
                    # This part needs careful handling if original page numbers were lost
                    # For now, we rely on pdfplumber adding it during initial parse
                    # and assume it's preserved if we re-parse the original PDF.
                    # A more robust solution would store page numbers in file_map_data.
                all_current_docs_for_rebuild.extend(docs)
            else:
                print(f"Warning: Text file not found for {f_info['original_filename']} ({f_id}). Skipping during rebuild.")
                # Optionally remove from file_map_data if file is missing
                # del file_map_data[f_id] 

        if not all_current_docs_for_rebuild:
            vectorstore = None # No documents left, clear vectorstore
            print("No documents remaining after rebuild, FAISS index cleared.")
        else:
            vectorstore = FAISS.from_documents(all_current_docs_for_rebuild, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)
            print(f"FAISS index rebuilt and saved to {FAISS_INDEX_PATH} with {len(all_current_docs_for_rebuild)} chunks.")

        save_file_map(file_map_data) # Save updated file map

        # Generate summary only for newly uploaded files
        summary = ""
        if full_text_for_summary.strip():
            summary = await summarization_chain.arun(text=full_text_for_summary)

    except Exception as e:
        print(f"Error during FAISS index rebuild or summary generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create or update the vector store: {e}")

    return {"message": f"Successfully added {len(newly_uploaded_files_info)} new file(s).", "summary": summary, "uploaded_files_info": newly_uploaded_files_info}





@app.post("/query")
async def query_documents(request: QueryRequest):
    global vectorstore
    if not vectorstore:
        return StreamingResponse(iter([json.dumps({"token": "The knowledge base is empty. Please upload documents first."})+"\n"]))

    session_id = request.session_id or str(uuid4())

    if session_id not in conversation_histories:
        conversation_histories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
    memory = conversation_histories[session_id]

    retriever = vectorstore.as_retriever()

    # Transform the query if there's chat history
    if memory.buffer:
        transformed_query = await query_transform_chain.arun(
            chat_history=memory.buffer_as_str,
            question=request.query
        )
        print(f"Transformed Query: {transformed_query}")
        question_to_use = transformed_query
    else:
        question_to_use = request.query

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_llm=llm, # Use LLM for condensing question
        combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_template(
            """Use the following context to answer the question. Be concise and answer in a single sentence if possible. If you don't know the answer, just say that you don't know.

            {context}

            Question: {question}
            """
        )}
    )

    async def stream_generator():
        full_response = ""
        try:
            result = await qa_chain.ainvoke({"question": question_to_use})
            full_response = result.get("answer", "")
            yield json.dumps({"token": full_response, "session_id": session_id}) + "\n"

            sources = result.get("source_documents", [])
            formatted_sources = []
            for source in sources:
                formatted_sources.append({
                    "content": source.page_content,
                    "metadata": source.metadata
                })
            yield json.dumps({"sources": formatted_sources}) + "\n"

        except Exception as e:
            print(f"Error during RAG chain execution: {e}")
            error_message = "An error occurred while generating the response."
            yield json.dumps({"token": error_message, "error": True}) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

