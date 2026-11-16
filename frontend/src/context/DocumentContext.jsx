import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import {
  getHealth,
  getDocuments,
  getDocumentStats,
  uploadDocument as apiUploadDocument,
  deleteDocument as apiDeleteDocument,
  FALLBACK_DOCUMENTS,
  FALLBACK_STATS,
} from '../services/api';

const DocumentContext = createContext(null);

export const DocumentProvider = ({ children }) => {
  const [documents, setDocuments] = useState(FALLBACK_DOCUMENTS);
  const [stats, setStats] = useState(FALLBACK_STATS);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [backendInfo, setBackendInfo] = useState(null);
  const [activeDocumentId, setActiveDocumentId] = useState(null);
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('ALL');
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [selectedDocForDetails, setSelectedDocForDetails] = useState(null);
  const [toast, setToast] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  // Show a temporary toast message
  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type, id: Date.now() });
    setTimeout(() => {
      setToast(null);
    }, 4500);
  }, []);

  // Fetch backend status and real document data
  const fetchData = useCallback(async () => {
    try {
      setBackendStatus('checking');
      const healthData = await getHealth();
      setBackendStatus('online');
      setBackendInfo(healthData);

      try {
        const docsData = await getDocuments();
        const loadedDocs = docsData?.documents || [];
        setDocuments(loadedDocs);

        // Maintain or update active document
        setActiveDocumentId((currentActive) => {
          if (loadedDocs.length === 0) return null;
          if (currentActive && loadedDocs.some((d) => d.id === currentActive)) {
            return currentActive;
          }
          return loadedDocs[0].id;
        });

        const statsData = await getDocumentStats();
        if (statsData) {
          setStats(statsData);
        }
      } catch (docErr) {
        console.warn('Could not load documents from backend:', docErr);
      }
    } catch (err) {
      console.warn('Backend currently offline or unreachable:', err.message);
      setBackendStatus('offline');
      setBackendInfo({
        status: 'offline',
        app: 'DocuMind AI',
        stage: 'Stage 2B (Local Offline Mode)',
        version: '1.0.0',
        error: err.message,
      });
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Poll backend health and data every 20 seconds
    const interval = setInterval(fetchData, 20000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Active selected document
  const activeDocument = documents.find((d) => d.id === activeDocumentId) || documents[0] || null;

  // Real upload function calling FastAPI POST /api/documents/upload
  const addDocument = async (file) => {
    setIsUploading(true);
    try {
      const createdDoc = await apiUploadDocument(file);
      await fetchData();
      setActiveDocumentId(createdDoc.id);
      showToast(`Uploaded and processed "${createdDoc.name || file.name}"`, 'success');
      return createdDoc;
    } catch (err) {
      const errMsg =
        err.response?.data?.detail ||
        err.message ||
        `Failed to upload "${file.name}".`;
      showToast(errMsg, 'error');
      throw err;
    } finally {
      setIsUploading(false);
    }
  };

  // Real delete function calling FastAPI DELETE /api/documents/{docId}
  const deleteDocument = async (docId) => {
    const docToDelete = documents.find((d) => d.id === docId);
    try {
      await apiDeleteDocument(docId);
      await fetchData();
      setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
      showToast(`Deleted "${docToDelete?.name || 'Document'}"`, 'info');
    } catch (err) {
      const errMsg =
        err.response?.data?.detail ||
        err.message ||
        `Failed to delete document.`;
      showToast(errMsg, 'error');
    }
  };

  // Toggle selection for Multi-Doc Q&A
  const toggleDocSelection = (docId) => {
    setSelectedDocIds((prev) => {
      if (prev.includes(docId)) {
        return prev.filter((id) => id !== docId);
      } else {
        return [...prev, docId];
      }
    });
  };

  return (
    <DocumentContext.Provider
      value={{
        documents,
        stats,
        backendStatus,
        backendInfo,
        activeDocument,
        activeDocumentId,
        setActiveDocumentId,
        selectedDocIds,
        setSelectedDocIds,
        toggleDocSelection,
        searchQuery,
        setSearchQuery,
        typeFilter,
        setTypeFilter,
        isUploadModalOpen,
        setIsUploadModalOpen,
        selectedDocForDetails,
        setSelectedDocForDetails,
        toast,
        showToast,
        addDocument,
        deleteDocument,
        isUploading,
        refreshData: fetchData,
      }}
    >
      {children}
    </DocumentContext.Provider>
  );
};

export const useDocuments = () => {
  const context = useContext(DocumentContext);
  if (!context) {
    throw new Error('useDocuments must be used within a DocumentProvider');
  }
  return context;
};
