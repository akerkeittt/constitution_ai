import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import tempfile
import re


st.set_page_config(page_title="Kazakhstan Constitution AI", layout="wide")
st.title("AI Assistant for Constitution of Kazakhstan")


llm = OllamaLLM(model="mistral")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


UPLOAD_DIR = tempfile.mkdtemp()


def load_documents(files):
    docs = []
    unsupported_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        if file.name.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            unsupported_files.append(file.name)
            continue
        docs.extend(loader.load())
    return docs, unsupported_files


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


def extract_article(full_text, article_number):
    pattern = rf"(?i)Article\s+{article_number}\s*(.*?)(?=\nArticle\s+\d+|$)"
    match = re.search(pattern, full_text, re.DOTALL)
    return match.group(0).strip() if match else None


if "vectordb" not in st.session_state:
    with open("constitution_kz.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    default_doc = Document(page_content=raw_text)
    chunks = split_documents([default_doc])
    vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db")
    st.session_state.vectordb = vectordb
    st.session_state.constitution_text = raw_text
    st.success("📘 Default Constitution loaded successfully.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header("📎 Upload Additional Documents")
uploaded_files = st.sidebar.file_uploader("Upload .txt, .pdf, or .docx files", accept_multiple_files=True)

if uploaded_files:
    docs, unsupported = load_documents(uploaded_files)
    if unsupported:
        st.sidebar.warning(f"Unsupported files skipped: {', '.join(unsupported)}")
    if docs:
        chunks = split_documents(docs)
        st.session_state.vectordb.add_documents(chunks)
        st.sidebar.success(f"Uploaded and indexed {len(uploaded_files) - len(unsupported)} file(s).")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about the Constitution of Kazakhstan...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    article_match = re.search(r"Article\s*(\d+)", user_input, re.IGNORECASE)

    if article_match:
        article_number = article_match.group(1)
        article_text = extract_article(st.session_state.constitution_text, article_number)

        if article_text:
            full_prompt = f"Answer this question using the following article:\n\n{article_text}\n\nQuestion: {user_input}"
            answer = llm.invoke(full_prompt)

            qa_doc = Document(page_content=f"Q: {user_input}\nA: {answer}")
            st.session_state.vectordb.add_documents([qa_doc])

            with st.chat_message("assistant"):
                st.markdown(f"**Answer:** {answer}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"**Answer:** {answer}"})
        else:
            warning_msg = "The specified article was not found in the loaded document."
            with st.chat_message("assistant"):
                st.warning(warning_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": warning_msg})
    else:
        retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa_chain({"query": user_input})
        answer = result["result"]

        qa_doc = Document(page_content=f"Q: {user_input}\nA: {answer}")
        st.session_state.vectordb.add_documents([qa_doc])
        with st.chat_message("assistant"):
            st.markdown(f"**Answer:** {answer}")
        st.session_state.chat_history.append({"role": "assistant", "content": f"**Answer:** {answer}"})
