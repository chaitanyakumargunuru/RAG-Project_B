import React, { useState } from 'react';

const API_BASE_URL = 'http://127.0.0.1:8000'; // Points to your FastAPI server running on Postgres

export default function App() {
  // Navigation & Auth State
  const [isLoginView, setIsLoginView] = useState(true);
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [error, setError] = useState('');

  // Form Inputs
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  // App States (Upload & Chat)
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploaded, setIsUploaded] = useState(false);
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hello! Upload a document above, and I will answer questions about it.' }
  ]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);

  // ==========================================
  // AUTHENTICATION HANDLERS
  // ==========================================
  const handleSignUp = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const response = await fetch(`${API_BASE_URL}/api/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ first_name: firstName, last_name: lastName, email, password }),
      });
      const data = await response.json();

      if (!response.ok) throw new Error(data.detail || 'Sign up failed');

      // Auto-switch to login view on success
      setIsLoginView(true);
      alert('Account created successfully! Please log in.');
    } catch (err) {
      setError(err.message);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    try {
      // FastAPI expects OAuth2 forms as application/x-www-form-urlencoded, not JSON!
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);

      const response = await fetch(`${API_BASE_URL}/api/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) throw new Error(data.detail || 'Invalid credentials');

      localStorage.setItem('token', data.access_token);
      setToken(data.access_token);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken('');
    setIsUploaded(false);
    setFile(null);
    setMessages([{ sender: 'bot', text: 'Hello! Upload a document above, and I will answer questions about it.' }]);
  };

  // ==========================================
  // DOCUMENT UPLOAD HANDLER
  // ==========================================
  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setUploadStatus('Uploading and processing vectors...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) throw new Error(data.detail || 'Upload failed');

      setUploadStatus('File successfully indexed in Vector DB!');
      setIsUploaded(true);
    } catch (err) {
      setUploadStatus(`Error: ${err.message}`);
    }
  };

  // ==========================================
  // CHAT / RAG HANDLER
  // ==========================================
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = { sender: 'user', text: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setLoading(true);

    try {
      // Note for Member C: This maps to the query/chat endpoint you will build.
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ question: query }),
      });
      const data = await response.json();

      setMessages((prev) => [...prev, { sender: 'bot', text: data.answer || 'No response from model.' }]);
    } catch (err) {
      setMessages((prev) => [...prev, { sender: 'bot', text: 'Error fetching response from RAG backend.' }]);
    } finally {
      setLoading(false);
    }
  };

  // ==========================================
  // RENDER VIEWS
  // ==========================================
  if (!token) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
        <div className="bg-white p-8 rounded-xl shadow-md w-full max-w-md">
          <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">
            {isLoginView ? 'Welcome Back' : 'Create Account'}
          </h2>

          {error && <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm mb-4">{error}</div>}

          <form onSubmit={isLoginView ? handleLogin : handleSignUp} className="space-y-4">
            {!isLoginView && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-gray-600 block mb-1">First Name</label>
                  <input type="text" required className="w-full border p-2 rounded-lg" value={firstName} onChange={e => setFirstName(e.target.value)} />
                </div>
                <div>
                  <label className="text-xs font-semibold text-gray-600 block mb-1">Last Name</label>
                  <input type="text" required className="w-full border p-2 rounded-lg" value={lastName} onChange={e => setLastName(e.target.value)} />
                </div>
              </div>
            )}
            <div>
              <label className="text-xs font-semibold text-gray-600 block mb-1">Email ID</label>
              <input type="email" required className="w-full border p-2 rounded-lg" value={email} onChange={e => setEmail(e.target.value)} />
            </div>
            <div>
              <label className="text-xs font-semibold text-gray-600 block mb-1">Password</label>
              <input type="password" required className="w-full border p-2 rounded-lg" value={password} onChange={e => setPassword(e.target.value)} />
            </div>
            <button type="submit" className="w-full bg-blue-600 text-white p-2 rounded-lg font-medium hover:bg-blue-700 transition">
              {isLoginView ? 'Sign In' : 'Register'}
            </button>
          </form>

          <p className="text-sm text-center text-gray-500 mt-6">
            {isLoginView ? "Don't have an account? " : "Already have an account? "}
            <button onClick={() => { setIsLoginView(!isLoginView); setError(''); }} className="text-blue-600 font-semibold underline">
              {isLoginView ? 'Sign Up' : 'Log In'}
            </button>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">

      <header className="bg-white border-b px-6 py-4 flex justify-between items-center shadow-sm">
        <h1 className="text-xl font-bold text-gray-800">📚 Simple RAG Application</h1>
        <button onClick={handleLogout} className="text-sm bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded-lg transition">
          Log Out
        </button>
      </header>


      <main className="flex-1 max-w-4xl w-full mx-auto p-6 flex flex-col gap-6">


        <div className="bg-white p-6 rounded-xl shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">1. Upload Context Document</h3>
          <form onSubmit={handleUpload} className="flex gap-4 items-center">
            <input
              type="file"
              accept=".pdf,.txt,.docx"
              onChange={(e) => setFile(e.target.files[0])}
              className="flex-1 text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            <button
              type="submit"
              disabled={!file}
              className="bg-blue-600 text-white px-5 py-2 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 transition"
            >
              Upload
            </button>
          </form>
          {uploadStatus && <p className="text-xs font-medium text-blue-600 mt-2">{uploadStatus}</p>}
        </div>


        <div className="flex-1 bg-white rounded-xl shadow-sm border flex flex-col overflow-hidden min-h-[400px]">
          <div className="bg-gray-50 border-b px-6 py-3 flex items-center justify-between">
            <h3 className="text-md font-semibold text-gray-800">2. Document Chatbot</h3>
            <span className={`text-xs px-2 py-1 rounded-full font-semibold ${isUploaded ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
              {isUploaded ? 'Document Loaded' : 'Waiting for Upload'}
            </span>
          </div>


          <div className="flex-1 p-6 space-y-4 overflow-y-auto max-h-[350px]">
            {messages.map((msg, index) => (
              <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[75%] rounded-xl px-4 py-2.5 text-sm ${
                  msg.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'
                }`}>
                  {msg.text}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-500 rounded-xl px-4 py-2.5 text-sm animate-pulse">
                  Thinking...
                </div>
              </div>
            )}
          </div>


          <form onSubmit={handleSendMessage} className="border-t p-4 bg-gray-50 flex gap-2">
            <input
              type="text"
              placeholder={isUploaded ? "Ask something about the document..." : "Please upload a document first..."}
              disabled={!isUploaded}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1 border px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
            />
            <button
              type="submit"
              disabled={!isUploaded || !query.trim() || loading}
              className="bg-blue-600 text-white px-5 py-2 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 transition"
            >
              Send
            </button>
          </form>
        </div>

      </main>
    </div>
  );
}