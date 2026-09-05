// Backend API base URL.
//
// Local development: temporarily set to "" -> app.js falls back to
// http://127.0.0.1:8000.
//
// Vercel / Render frontend deployment: points at the deployed backend so
// production never falls back to a loopback address.
window.RAG_API_BASE = "https://chatbot-exuw.onrender.com";