# DocuMind

DocuMind is an intelligent document analysis and question-answering system. It allows you to upload documents in various formats (PDF, DOCX, TXT) and then ask questions about their content. The system uses a language model to understand the documents and provide relevant answers.

## Features

*   **Document Upload:** Upload multiple documents in PDF, DOCX, and TXT formats.
*   **Question Answering:** Ask questions about the uploaded documents and get intelligent answers.
*   **Summarization:** Get a summary of the uploaded documents.
*   **Source Highlighting:** See the sources from the documents that were used to generate the answer.
*   **Conversation History:** The system remembers the context of the conversation, allowing for follow-up questions.
*   **Knowledge Base Management:** Clear the knowledge base to start fresh with new documents.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Python 3.7+
*   Node.js and npm

### Installation

1.  **Clone the repo**
    ```sh
    git clone https://github.com/kesav2003/DocuMind.git
    ```

2.  **Backend Setup**
    ```sh
    cd backend
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    cp .env.example .env
    # Add your PINECONE_API_KEY and other environment variables to the .env file
    uvicorn main:app --reload
    ```

3.  **Frontend Setup**
    ```sh
    cd frontend
    npm install
    npm run dev
    ```

## Usage

1.  **Upload Documents:** Use the "Choose Files" button to select the documents you want to analyze.
2.  **Ask Questions:** Once the documents are uploaded, you can ask questions in the input field and get answers from the AI.
3.  **View Sources:** Click on the "Sources" button to see the parts of the documents that were used to generate the answer.
4.  **Clear Knowledge Base:** Use the "Clear All" button to remove all uploaded documents and start over.

## Evaluation

The system is evaluated using a set of questions and ground truth answers from a dataset of news articles and recipes. The evaluation script (`backend/evaluation.py`) calculates the following metrics:

*   **ROUGE-1:** Measures the overlap of unigrams between the generated answer and the ground truth answer.
*   **BERTScore:** Measures the similarity between the generated answer and the ground truth answer using BERT embeddings.
*   **BLEU:** Measures the similarity between the generated answer and the ground truth answer based on n-gram precision.

## Technologies Used

### Backend

*   **Python**
*   **FastAPI:** A modern, fast (high-performance) web framework for building APIs with Python 3.7+.
*   **LangChain:** A framework for developing applications powered by language models.
*   **Ollama:** A tool for running large language models locally.
*   **FAISS:** A library for efficient similarity search and clustering of dense vectors.
*   **pdfplumber:** A library for extracting text from PDF files.
*   **python-docx:** A library for creating and updating Microsoft Word (.docx) files.

### Frontend

*   **React:** A JavaScript library for building user interfaces.
*   **Vite:** A fast frontend build tool.
*   **axios:** A promise-based HTTP client for the browser and Node.js.
