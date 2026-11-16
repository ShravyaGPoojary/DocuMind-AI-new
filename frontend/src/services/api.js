import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180000, // 180s (3m) timeout for local LLM generation and document processing
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

// Attach JWT token to all requests if present in localStorage
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('documind_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);


// Fallback documents (empty by default)
export const FALLBACK_DOCUMENTS = [];

export const FALLBACK_STATS = {
  total_documents: 0,
  processed_documents: 0,
  processing_documents: 0,
  total_storage_mb: 0,
  total_chunks: 0,
  system_status: 'healthy',
};

/**
 * Health check endpoint
 */
export const getHealth = async () => {
  const startTime = Date.now();
  const response = await apiClient.get('/health');
  const latency = Date.now() - startTime;
  return { ...response.data, latency };
};

/**
 * Project info endpoint
 */
export const getProjectInfo = async () => {
  const response = await apiClient.get('/');
  return response.data;
};

/**
 * Documents listing endpoint
 */
export const getDocuments = async (skip = 0, limit = 100) => {
  const response = await apiClient.get('/documents', {
    params: { skip, limit },
  });
  return response.data;
};

/**
 * Document statistics endpoint
 */
export const getDocumentStats = async () => {
  const response = await apiClient.get('/documents/stats');
  return response.data;
};

/**
 * Upload and process document endpoint (multipart/form-data)
 */
export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/documents/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

/**
 * Single document details endpoint
 */
export const getDocument = async (documentId) => {
  const response = await apiClient.get(`/documents/${documentId}`);
  return response.data;
};

/**
 * Document chunks endpoint
 */
export const getDocumentChunks = async (documentId, page = 1, pageSize = 50) => {
  const skip = (page - 1) * pageSize;
  const response = await apiClient.get(`/documents/${documentId}/chunks`, {
    params: { skip, limit: pageSize },
  });
  return response.data;
};

/**
 * Delete document endpoint
 */
export const deleteDocument = async (documentId) => {
  const response = await apiClient.delete(`/documents/${documentId}`);
  return response.data;
};

/**
 * Generate vector embeddings for a specific document's chunks
 */
export const embedDocument = async (documentId) => {
  const response = await apiClient.post(`/documents/${documentId}/embed`);
  return response.data;
};

/**
 * Generate vector embeddings for all eligible documents
 */
export const embedAllDocuments = async () => {
  const response = await apiClient.post('/documents/embed-all');
  return response.data;
};

/**
 * Check embedding status for a document
 */
export const getDocumentEmbeddingStatus = async (documentId) => {
  const response = await apiClient.get(`/documents/${documentId}/embedding-status`);
  return response.data;
};

/**
 * Get ChromaDB and embedding vector statistics
 */
export const getEmbeddingStats = async () => {
  const response = await apiClient.get('/embeddings/stats');
  return response.data;
};

/**
 * Execute semantic vector search across document chunks
 */
export const searchDocuments = async (query, topK = 5, documentId = null, documentIds = null) => {
  const payload = {
    query,
    top_k: topK,
  };
  if (documentIds && Array.isArray(documentIds) && documentIds.length > 0) {
    payload.document_ids = documentIds;
  } else if (documentId) {
    payload.document_id = documentId;
  }
  const response = await apiClient.post('/search', payload);
  return response.data;
};

/**
 * Generate completion text using the default (Groq) or optional (Ollama) LLM provider
 */
export const generateLLMResponse = async (
  prompt,
  temperature = 0.2,
  maxTokens = 1024,
  systemPrompt = null,
  model = null,
  provider = null
) => {
  const payload = {
    prompt,
    temperature,
    max_tokens: maxTokens,
  };
  if (systemPrompt) payload.system_prompt = systemPrompt;
  if (model) payload.model = model;
  if (provider) payload.provider = provider;
  const response = await apiClient.post('/llm/generate', payload);
  return response.data;
};

/**
 * Get LLM health status (Groq and Ollama)
 */
export const getLLMHealth = async () => {
  const response = await apiClient.get('/llm/health');
  return response.data;
};

/**
 * List supported models from default or specified LLM provider
 */
export const getLLMModels = async (provider = null) => {
  const response = await apiClient.get('/llm/models', {
    params: provider ? { provider } : {},
  });
  return response.data;
};

/**
 * Send question to RAG pipeline for grounded document answer
 */
export const sendChatMessage = async ({
  question,
  documentId = null,
  documentIds = null,
  topK = 4,
  conversationId = null,
  provider = null,
  model = null,
  timeout = 180000,
}) => {
  const payload = {
    question,
    top_k: topK,
  };
  if (documentIds && Array.isArray(documentIds) && documentIds.length > 0) {
    payload.document_ids = documentIds;
  } else if (documentId) {
    payload.document_id = documentId;
  }
  if (conversationId) payload.conversation_id = conversationId;
  if (provider) payload.provider = provider;
  if (model) payload.model = model;
  const response = await apiClient.post('/chat', payload, {
    timeout,
  });
  return response.data;
};

/**
 * List saved conversation threads
 */
export const getConversations = async (skip = 0, limit = 50) => {
  const response = await apiClient.get('/chat/conversations', {
    params: { skip, limit },
  });
  return response.data;
};

/**
 * Get conversation history by ID
 */
export const getConversation = async (conversationId) => {
  const response = await apiClient.get(`/chat/conversations/${conversationId}`);
  return response.data;
};

/**
 * Delete conversation by ID
 */
export const deleteConversation = async (conversationId) => {
  const response = await apiClient.delete(`/chat/conversations/${conversationId}`);
  return response.data;
};

/**
 * Generate intelligent document summary (Groq GPT-OSS 120B default, synchronous)
 */
export const generateDocumentSummary = async (
  documentId,
  { summaryType = 'brief', provider = null, model = null, timeout = 180000 } = {}
) => {
  const payload = {
    summary_type: summaryType,
  };
  if (provider) payload.provider = provider;
  if (model) payload.model = model;
  const response = await apiClient.post(`/summarization/${documentId}`, payload, {
    timeout,
  });
  return response.data;
};

/**
 * Start asynchronous/background document summary job with real-time progress tracking
 */
export const startDocumentSummaryJob = async (
  documentId,
  { summaryType = 'brief', provider = null, model = null } = {}
) => {
  const payload = {
    summary_type: summaryType,
    background: true,
  };
  if (provider) payload.provider = provider;
  if (model) payload.model = model;
  const response = await apiClient.post(`/summarization/${documentId}`, payload, {
    params: { background: true },
  });
  return response.data;
};

/**
 * Poll real-time status and progress for an active summarization job
 */
export const getSummaryJobStatus = async (jobId) => {
  const response = await apiClient.get(`/summarization/jobs/${jobId}`);
  return response.data;
};

/**
 * Resume a paused, rate-limited, or failed summarization job
 */
export const resumeSummaryJob = async (jobId) => {
  const response = await apiClient.post(`/summarization/jobs/${jobId}/resume`);
  return response.data;
};

/**
 * Get supported summary types catalog
 */
export const getSummaryTypes = async () => {
  const response = await apiClient.get('/summarization/types');
  return response.data;
};

/**
 * Compare two documents and perform cross-document analysis
 */
export const compareDocuments = async (documentId1, documentId2, focusArea = null, provider = 'groq', model = null) => {
  const payload = {
    document_id_1: documentId1,
    document_id_2: documentId2,
    provider,
  };
  if (focusArea && focusArea.trim()) {
    payload.focus_area = focusArea.trim();
  }
  if (model && model.trim()) {
    payload.model = model.trim();
  }
  const response = await apiClient.post('/comparison/compare', payload);
  return response.data;
};

/* ==========================================================================
   Authentication & User Management APIs (Stage 8)
   ========================================================================== */

/**
 * Register a new user account with full name, email, and password
 */
export const registerUser = async (fullName, email, password) => {
  const response = await apiClient.post('/auth/register', {
    full_name: fullName,
    email,
    password,
  });
  return response.data;
};

/**
 * Login user with email and password
 */
export const loginUser = async (email, password) => {
  const response = await apiClient.post('/auth/login', {
    email,
    password,
  });
  return response.data;
};

/**
 * Sign in or register with verified Google ID token
 */
export const googleAuth = async (credential) => {
  const response = await apiClient.post('/auth/google', {
    credential,
  });
  return response.data;
};

/**
 * Get profile of currently authenticated user
 */
export const getCurrentUser = async () => {
  const response = await apiClient.get('/auth/me');
  return response.data;
};

/**
 * Log out user
 */
export const logoutUser = async () => {
  const response = await apiClient.post('/auth/logout');
  return response.data;
};

/* ==========================================================================
   Advanced Document Intelligence APIs (Stage 8)
   ========================================================================== */

/**
 * Extract structured entities, dates, definitions, and metrics
 */
export const extractInformation = async (documentId) => {
  const response = await apiClient.post(`/intelligence/extract/${documentId}`);
  return response.data;
};

/**
 * Generate deep strategic insights, roadmap, and major findings
 */
export const getDocumentInsights = async (documentId) => {
  const response = await apiClient.post(`/intelligence/insights/${documentId}`);
  return response.data;
};

/**
 * Generate knowledge-check revision questions and answers
 */
export const generateQuestions = async (documentId, numQuestions = 5, difficulty = 'mixed') => {
  const response = await apiClient.post(`/intelligence/generate-questions/${documentId}`, {
    num_questions: numQuestions,
    difficulty,
  });
  return response.data;
};

/**
 * Perform multi-hop contextual reasoning on a specific inquiry or hypothesis
 */
export const contextualAnalysis = async (documentId, inquiry) => {
  const response = await apiClient.post(`/intelligence/contextual-analysis/${documentId}`, {
    inquiry,
  });
  return response.data;
};

export default {
  getHealth,
  getProjectInfo,
  getDocuments,
  getDocumentStats,
  uploadDocument,
  getDocument,
  getDocumentChunks,
  deleteDocument,
  embedDocument,
  embedAllDocuments,
  getDocumentEmbeddingStatus,
  getEmbeddingStats,
  searchDocuments,
  generateLLMResponse,
  getLLMHealth,
  getLLMModels,
  sendChatMessage,
  getConversations,
  getConversation,
  deleteConversation,
  generateDocumentSummary,
  startDocumentSummaryJob,
  getSummaryJobStatus,
  resumeSummaryJob,
  getSummaryTypes,
  compareDocuments,
  registerUser,
  loginUser,
  googleAuth,
  getCurrentUser,
  logoutUser,
  extractInformation,
  getDocumentInsights,
  generateQuestions,
  contextualAnalysis,
};


