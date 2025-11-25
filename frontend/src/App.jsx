import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid'; // For unique message keys
import './App.css';

function Sources({ sources }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="sources-container">
      <button onClick={() => setIsOpen(!isOpen)} className="sources-button">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="feather feather-link">
          <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.72"></path>
          <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.72-1.72"></path>
        </svg>
      </button>
      {isOpen && (
        <div className="sources-list">
          <h3>Sources:</h3>
          {sources.map((source, index) => (
            <div key={index} className="source-item">
              <p><strong>Source {index + 1}:</strong> {source.metadata.source}{source.metadata.page ? ` (Page ${source.metadata.page})` : ''}</p>
              <pre>{source.content}</pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const API_BASE_URL = 'http://127.0.0.1:8000';

function App() {
  const [query, setQuery] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const [sessionId, setSessionId] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [showFileList, setShowFileList] = useState(false); // New state for toggling file list

  // Use a ref to track if initial fetch has happened
  const hasFetchedInitialFiles = useRef(false);

  const fetchUploadedFiles = useCallback(async () => {
    // No longer need the hasFetchedInitialFiles check if we want to be able to refresh
    try {
      const response = await axios.get(`${API_BASE_URL}/list_files`);
      setUploadedFiles(response.data.files);
    } catch (err) {
      console.error("Error fetching uploaded files:", err);
      setError("Error fetching uploaded files.");
    }
  }, [setUploadedFiles, setError]);

  // Fetch files on initial render
  useEffect(() => {
    fetchUploadedFiles();
  }, [fetchUploadedFiles]);


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleQueryChange = (e) => { setQuery(e.target.value); };
  const handleFileChange = (e) => { setSelectedFiles(Array.from(e.target.files)); };

  const handleRemoveFile = (indexToRemove) => {
    setSelectedFiles(prevFiles => prevFiles.filter((_, index) => index !== indexToRemove));
  };

  const handleClear = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/clear`);
      setMessages([{
        id: uuidv4(),
        sender: 'system',
        text: response.data.message || 'Knowledge base cleared. Ready for new documents.'
      }]);
      fetchUploadedFiles(); // Refresh the list of uploaded files
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Error clearing knowledge base.';
      setError(errorMsg);
      console.error(err);
    } finally {
      setLoading(false);
      setSelectedFiles([]);
      setQuery('');
    }
  };

  const handleDeleteFile = async (fileId, filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await axios.delete(`${API_BASE_URL}/delete_file/${fileId}`);
      setMessages(prev => [...prev, { id: uuidv4(), sender: 'system', text: response.data.message }]);
      fetchUploadedFiles(); // Refresh the list of uploaded files
    } catch (err) {
      const errorMsg = err.response?.data?.detail || `Error deleting file ${filename}.`;
      setError(errorMsg);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select one or more files to upload.');
      return;
    }
    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });

    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setMessages(prev => [...prev, { id: uuidv4(), sender: 'system', text: response.data.message }]);
      if (response.data.summary) {
        setMessages(prev => [...prev, { id: uuidv4(), sender: 'ai', text: response.data.summary }]);
      }
      setSelectedFiles([]); // Clear selection after upload
      fetchUploadedFiles(); // Refresh the list of uploaded files
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Error uploading files.';
      setError(errorMsg);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSendQuery = async () => {
    if (!query.trim()) return;

    const userMessage = { id: uuidv4(), sender: 'user', text: query };
    const aiMessageId = uuidv4();
    const aiMessagePlaceholder = { id: aiMessageId, sender: 'ai', text: '' };

    setMessages(prev => [...prev, userMessage, aiMessagePlaceholder]);
    setQuery('');
    setError(null);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: sessionId })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.lastIndexOf('\n');
        if (boundary === -1) continue;

        const completeLines = buffer.substring(0, boundary);
        buffer = buffer.substring(boundary + 1);

        const lines = completeLines.split('\n').filter(line => line.trim() !== '');
        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.token) {
                fullResponse += data.token;
            }
            if (data.session_id) {
              setSessionId(data.session_id);
            }
            if (data.sources) {
                setMessages(prev => prev.map(msg =>
                    msg.id === aiMessageId ? { ...msg, sources: data.sources } : msg
                ));
            } else {
                setMessages(prev => prev.map(msg =>
                    msg.id === aiMessageId ? { ...msg, text: fullResponse } : msg
                ));
            }
          } catch (e) {
            console.error("Error parsing chunk: ", line);
          }
        }
      }
    } catch (err) {
      const errorMsg = err.message || 'Error getting response.';
      setError(errorMsg);
      setMessages(prev => prev.filter(msg => msg.id !== aiMessageId)); // Remove placeholder on error
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>DocuMind</h1>
        <p>Your Personal AI Document Assistant</p>
      </header>

      <main className="chat-container">
        <div className="chat-messages">
          {messages.map((msg) => (
            <div key={msg.id} className="message-container">
              <div className={`message ${msg.sender}`}>
                {msg.text}
              </div>
              {msg.sender === 'ai' && <Sources sources={msg.sources} />}
            </div>
          ))}
          {error && <div className="message error">Error: {error}</div>}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-area">
          <input
            type="text"
            placeholder="Ask a question about your documents..."
            value={query}
            onChange={handleQueryChange}
            onKeyPress={(e) => e.key === 'Enter' && !loading && handleSendQuery()}
            disabled={loading}
          />
          <button onClick={handleSendQuery} disabled={loading}>
            {loading ? '... ' : 'Send'}
          </button>
        </div>
      </main>

      <footer className="app-footer">
        <div className="file-upload-area">
          <label htmlFor="file-upload" className="file-input-label">
            Choose Files
          </label>
          <input
            id="file-upload"
            type="file"
            multiple
            onChange={handleFileChange}
            disabled={loading}
          />
          <div className="selected-files-list">
            {selectedFiles.length > 0 ? (
              selectedFiles.map((file, index) => (
                <div key={file.name + index} className="selected-file-item">
                  <span>{file.name}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveFile(index)}
                    className="remove-file-button"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-x">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
                  </button>
                </div>
              ))
            ) : (
              <span className="file-names">
                No files chosen
              </span>
            )}
          </div>
        </div>
        <div className="footer-buttons">
          <button onClick={handleUpload} className="upload-btn" disabled={loading || selectedFiles.length === 0}>
            Upload
          </button>
          <button onClick={handleClear} className="clear-btn" disabled={loading}>
            Clear All
          </button>
        </div>
      </footer>

      <div className="uploaded-files-display">
        <div className="uploaded-files-header">
          <h3>Uploaded Documents:</h3>
          <button onClick={() => setShowFileList(!showFileList)} className="toggle-files-btn">
            {showFileList ? 'Hide' : 'Show'}
          </button>
        </div>
        {showFileList && (
          uploadedFiles.length > 0 ? (
            <ul>
              {uploadedFiles.map(file => (
                <li key={file.id}>
                  {file.filename}
                  <button
                    type="button"
                    onClick={() => handleDeleteFile(file.id, file.filename)}
                    className="delete-uploaded-file-button"
                    disabled={loading}
                  >
                    &times;
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p>No documents uploaded yet.</p>
          )
        )}
      </div>
    </div>
  );
}

export default App;