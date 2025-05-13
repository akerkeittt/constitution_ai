# 🇰🇿 AI Assistant for Constitution of Kazakhstan

An interactive Streamlit application that allows users to upload legal documents and ask questions about the **Constitution of Kazakhstan** using **natural language processing (NLP)** and **retrieval-based AI**. Built using `LangChain`, `Ollama LLM`, `HuggingFace Embeddings`, and `Chroma`.

---

## 🚀 Usage

1. **Install dependencies**:
    ```bash
    python -m pip install -r requirements.txt
    ```

2. **Run the app**:
    ```bash
    streamlit run app.py
    ```

3. **What you can do**:
    - Ask questions like:  
      `What is stated in Article 5?`  
      `What are the rights of citizens?`
    - Upload additional `.txt`, `.pdf`, or `.docx` files via the sidebar.
    - The assistant answers using:
        - Relevant constitutional articles if mentioned
        - Vector similarity search (via ChromaDB) if no article specified
    - Chat history is preserved during the session.

---

## 🖼️ Demo Screenshots

![image](https://github.com/user-attachments/assets/2b65ff42-8398-4fde-a881-04f7c19fbd33)

![image](https://github.com/user-attachments/assets/69ee0a12-eded-40e0-ba36-9852054d8ea6)

![image](https://github.com/user-attachments/assets/27988ec3-d2c3-4321-9bfb-bfc1f72002c7)

![image](https://github.com/user-attachments/assets/af2920e4-5546-49fa-ad2f-1535d7ebdeba)

![image](https://github.com/user-attachments/assets/4103c655-ba57-4b65-af17-9c82ea9295a8)


---

## 📁 Project Structure

📦 constitution-ai-assistant
┣ 📄 app.py                 # Main Streamlit app
┣ 📄 constitution_kz.txt    # Core legal document
┣ 📁 chroma_db/             # Vector store directory (auto-generated)
┣ 📄 requirements.txt       # Dependencies
┗ 📄 README.md              # This file

---

## 📝 License

This project is licensed under the MIT License.  
See [LICENSE](https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/LICENSE) for details.

---
