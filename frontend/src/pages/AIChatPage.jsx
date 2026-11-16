import React, { useState, useRef, useEffect } from 'react';
import {
  MessageSquare,
  Send,
  Sparkles,
  Bot,
  User,
  Bookmark,
  Trash2,
  Plus,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  FileText,
  Layers,
  Boxes,
  CheckSquare,
  Square,
  X,
} from 'lucide-react';
import Button from '../components/common/Button';
import { FileTypeBadge } from '../components/common/Badge';
import EmptyState from '../components/common/EmptyState';
import { useDocuments } from '../context/DocumentContext';
import { sendChatMessage } from '../services/api';

const DEFAULT_SUGGESTIONS = [
  'What are the key takeaways from these documents?',
  'Can you summarize the main findings or methodology across the files?',
  'What conclusions or actionable items are mentioned?',
];

const AIChatPage = () => {
  const {
    documents,
    activeDocument,
    activeDocumentId,
    setActiveDocumentId,
    selectedDocIds,
    setSelectedDocIds,
    toggleDocSelection,
    setIsUploadModalOpen,
  } = useDocuments();

  const [messages, setMessages] = useState([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [conversationId, setConversationId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [expandedSources, setExpandedSources] = useState({});
  const [isTargetPopoverOpen, setIsTargetPopoverOpen] = useState(false);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const popoverRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Close target dropdown popover on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target)) {
        setIsTargetPopoverOpen(false);
      }
    };
    if (isTargetPopoverOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isTargetPopoverOpen]);

  // Toggle display of sources for a specific message
  const toggleSourceExpansion = (msgId) => {
    setExpandedSources((prev) => ({
      ...prev,
      [msgId]: !prev[msgId],
    }));
  };

  // Determine current document target scope
  const isMultiSelected = selectedDocIds && selectedDocIds.length > 1;
  const isSingleCustomSelected = selectedDocIds && selectedDocIds.length === 1;
  const isAllDocsMode = !selectedDocIds || selectedDocIds.length === 0;

  const activeSelectedDocs = documents.filter((d) => selectedDocIds.includes(d.id));

  const handleSendMessage = async (textToSend) => {
    const query = (textToSend || inputPrompt).trim();
    if (!query || isLoading) return;

    setErrorMessage(null);

    const userMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputPrompt('');
    setIsLoading(true);

    try {
      let targetDocId = null;
      let targetDocIds = null;

      if (selectedDocIds && selectedDocIds.length > 0) {
        targetDocIds = selectedDocIds;
        if (selectedDocIds.length === 1) {
          targetDocId = selectedDocIds[0];
        }
      } else if (activeDocumentId && activeDocumentId !== 'all') {
        targetDocId = activeDocumentId;
        targetDocIds = [activeDocumentId];
      }

      const topK =
        targetDocIds && targetDocIds.length > 1
          ? Math.max(4, Math.min(12, targetDocIds.length * 3))
          : 4;

      const response = await sendChatMessage({
        question: query,
        documentId: targetDocId,
        documentIds: targetDocIds,
        topK,
        conversationId,
        timeout: 180000,
      });

      if (response.conversation_id) {
        setConversationId(response.conversation_id);
      }

      const assistantMessage = {
        id: `asst-${Date.now()}`,
        sender: 'assistant',
        text: response.answer,
        sources: response.sources || [],
        model: response.model,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error('RAG Chat request failed:', err);
      let errorDetail =
        err.response?.data?.detail ||
        err.message ||
        'Failed to connect to RAG service.';

      if (err.response?.data?.detail) {
        errorDetail = err.response.data.detail;
      } else if (err.code === 'ECONNABORTED' || (err.message && err.message.toLowerCase().includes('timeout'))) {
        errorDetail =
          'The AI model timed out while generating a response. Try asking a more specific question.';
      } else if (err.response?.status === 504) {
        errorDetail =
          'LLM generation timed out on the backend. Please try again with a more focused question.';
      } else if (err.response?.status === 503) {
        errorDetail =
          'Service is temporarily unavailable. Please verify network connectivity and backend service status.';
      } else if (err.response?.status === 401) {
        errorDetail =
          'Authentication failed. Please verify your GROQ_API_KEY in backend/.env.';
      } else if (err.response?.status === 429) {
        errorDetail =
          'Groq API rate limit reached. Please wait a moment before sending another query.';
      }

      setErrorMessage(errorDetail);

      const errorAssistantMsg = {
        id: `asst-err-${Date.now()}`,
        sender: 'assistant',
        isError: true,
        text: `Error: ${errorDetail}`,
        sources: [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorAssistantMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setConversationId(null);
    setErrorMessage(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (documents.length === 0) {
    return (
      <div className="chat-page" style={{ padding: '40px 24px', justifyContent: 'center' }}>
        <EmptyState
          icon={MessageSquare}
          title="No documents uploaded yet"
          description="Upload documents to ask questions and receive grounded answers with citations."
          actionText="Upload Document"
          actionIcon={Plus}
          onAction={() => setIsUploadModalOpen(true)}
        />
      </div>
    );
  }

  return (
    <div className="chat-page">
      {/* Top Document Selector Bar */}
      <div className="chat-header-bar">
        <div className="chat-doc-selector-group">
          <span className="selector-label">Chat Target:</span>

          {/* Interactive Target Popover Trigger */}
          <div className="chat-target-selector-container" ref={popoverRef}>
            <button
              type="button"
              className={`chat-target-trigger-btn ${selectedDocIds.length > 0 ? 'active' : ''}`}
              onClick={() => setIsTargetPopoverOpen(!isTargetPopoverOpen)}
              disabled={isLoading}
            >
              {isAllDocsMode ? (
                <>
                  <Layers size={14} />
                  <span>Entire Document Library ({documents.length} files)</span>
                </>
              ) : isSingleCustomSelected ? (
                <>
                  <FileText size={14} />
                  <span className="truncate" style={{ maxWidth: '200px' }}>
                    {activeSelectedDocs[0]?.name || '1 Document'}
                  </span>
                </>
              ) : (
                <>
                  <Boxes size={14} />
                  <span>{selectedDocIds.length} Documents Selected</span>
                </>
              )}
              <ChevronDown size={14} />
            </button>

            {/* Dropdown Popover */}
            {isTargetPopoverOpen && (
              <div className="chat-target-popover">
                <div className="chat-target-popover-header">
                  <span style={{ fontWeight: 600, fontSize: '0.8125rem' }}>Select Target Scope</span>
                  <div className="chat-target-actions">
                    <button
                      type="button"
                      className="chat-target-action-btn"
                      onClick={() => {
                        setSelectedDocIds(documents.map((d) => d.id));
                      }}
                    >
                      Select All
                    </button>
                    <span style={{ color: 'var(--border-strong)' }}>|</span>
                    <button
                      type="button"
                      className="chat-target-action-btn"
                      onClick={() => {
                        setSelectedDocIds([]);
                        setActiveDocumentId('all');
                      }}
                    >
                      Clear (All Docs)
                    </button>
                  </div>
                </div>

                <div
                  className={`chat-target-item ${isAllDocsMode ? 'selected' : ''}`}
                  onClick={() => {
                    setSelectedDocIds([]);
                    setActiveDocumentId('all');
                    setIsTargetPopoverOpen(false);
                  }}
                >
                  <Layers size={14} color="var(--primary-600)" />
                  <span style={{ flex: 1, fontWeight: isAllDocsMode ? 600 : 400 }}>
                    Entire Document Library ({documents.length} files)
                  </span>
                </div>

                <div style={{ height: '1px', backgroundColor: 'var(--border-light)', margin: '2px 0' }} />

                <div className="chat-target-items-list">
                  {documents.map((doc) => {
                    const isChecked = selectedDocIds.includes(doc.id);
                    return (
                      <div
                        key={doc.id}
                        className={`chat-target-item ${isChecked ? 'selected' : ''}`}
                        onClick={() => toggleDocSelection(doc.id)}
                      >
                        {isChecked ? (
                          <CheckSquare size={14} color="var(--primary-600)" />
                        ) : (
                          <Square size={14} color="var(--border-strong)" />
                        )}
                        <span className="truncate" style={{ flex: 1 }} title={doc.name}>
                          {doc.name}
                        </span>
                        <span style={{ fontSize: '0.6875rem', color: 'var(--text-tertiary)' }}>
                          {doc.file_type}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Quick status indicator */}
          {isAllDocsMode ? (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '0.75rem',
                color: 'var(--primary-700)',
                backgroundColor: 'var(--primary-50)',
                padding: '2px 8px',
                borderRadius: 'var(--radius-full)',
                fontWeight: 500,
              }}
            >
              <Layers size={12} />
              Cross-Document Search
            </span>
          ) : isSingleCustomSelected && activeSelectedDocs[0] ? (
            <FileTypeBadge type={activeSelectedDocs[0].file_type} />
          ) : (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '0.75rem',
                color: 'var(--primary-700)',
                backgroundColor: 'var(--primary-50)',
                padding: '2px 8px',
                borderRadius: 'var(--radius-full)',
                fontWeight: 500,
              }}
            >
              <Boxes size={12} />
              Balanced Multi-Doc Search
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {messages.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              icon={Trash2}
              onClick={handleClearChat}
              disabled={isLoading}
              title="Clear conversation"
            >
              New Chat
            </Button>
          )}
        </div>
      </div>

      {/* Selected Document Chips Bar (shown when custom documents are selected) */}
      {!isAllDocsMode && activeSelectedDocs.length > 0 && (
        <div className="chat-doc-chips-bar">
          <span style={{ color: 'var(--text-tertiary)', fontWeight: 500 }}>Active files:</span>
          {activeSelectedDocs.map((doc) => (
            <span key={doc.id} className="chat-doc-chip" title={doc.name}>
              <FileText size={11} />
              <span className="truncate">{doc.name}</span>
              <button
                type="button"
                className="chat-doc-chip-remove"
                onClick={() => toggleDocSelection(doc.id)}
                title="Remove file from chat scope"
              >
                <X size={12} />
              </button>
            </span>
          ))}
          <button
            type="button"
            className="chat-target-action-btn"
            style={{ marginLeft: '4px', fontSize: '0.6875rem' }}
            onClick={() => {
              setSelectedDocIds([]);
              setActiveDocumentId('all');
            }}
          >
            Clear All
          </button>
        </div>
      )}

      {/* Global Error Banner */}
      {errorMessage && (
        <div className="chat-error-banner">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertCircle size={16} />
            <span>{errorMessage}</span>
          </div>
          <Button variant="ghost" size="sm" onClick={() => setErrorMessage(null)}>
            Dismiss
          </Button>
        </div>
      )}

      {/* Messages Thread Container */}
      <div className="chat-messages-container">
        {messages.length === 0 ? (
          <div className="chat-welcome-box">
            <div className="welcome-icon-box">
              <Bot size={28} />
            </div>
            <h2 className="welcome-title">Ask Questions Grounded in Your Documents</h2>
            <p className="welcome-description">
              {isAllDocsMode ? (
                <>
                  Searching across all <strong>{documents.length}</strong> uploaded documents.
                  DocuMind AI will retrieve relevant excerpts and synthesize truthful answers
                  using Groq GPT-OSS 120B.
                </>
              ) : isSingleCustomSelected && activeSelectedDocs[0] ? (
                <>
                  Focused on <strong>{activeSelectedDocs[0].name}</strong>. Questions will be answered
                  strictly using chunks from this document.
                </>
              ) : (
                <>
                  Focused on <strong>{activeSelectedDocs.length} selected documents</strong>. Questions will be answered
                  with balanced cross-document retrieval and source attribution.
                </>
              )}
            </p>

            <span className="suggested-questions-title">Example questions:</span>
            <div className="suggested-questions-grid">
              {DEFAULT_SUGGESTIONS.map((q, idx) => (
                <div
                  key={idx}
                  className="suggested-question-pill"
                  style={{
                    opacity: isLoading ? 0.6 : 1,
                    cursor: isLoading ? 'not-allowed' : 'pointer',
                  }}
                  onClick={() => !isLoading && handleSendMessage(q)}
                >
                  <span>{q}</span>
                  <Send size={13} style={{ color: 'var(--primary-500)' }} />
                </div>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => {
            const isUser = msg.sender === 'user';
            const hasSources = msg.sources && msg.sources.length > 0;
            const isExpanded = !!expandedSources[msg.id];

            const uniqueSourcesDocCount = hasSources
              ? new Set(msg.sources.map((s) => s.document_id || s.filename)).size
              : 0;

            return (
              <div key={msg.id} className={`message-row ${msg.sender}`}>
                <div
                  className={`message-avatar ${
                    isUser ? 'user-avatar-icon' : 'assistant-avatar-icon'
                  }`}
                >
                  {isUser ? <User size={16} /> : <Bot size={16} />}
                </div>

                <div
                  className="message-bubble"
                  style={{
                    backgroundColor: msg.isError ? '#fef2f2' : undefined,
                    borderColor: msg.isError ? '#fecaca' : undefined,
                    color: msg.isError ? '#991b1b' : undefined,
                  }}
                >
                  <div
                    style={{
                      fontWeight: isUser ? 400 : 450,
                      whiteSpace: 'pre-wrap',
                      lineHeight: 1.6,
                    }}
                  >
                    {msg.text}
                  </div>

                  {/* Grounded Source Citations Accordion */}
                  {hasSources && (
                    <div className="rag-sources-section">
                      <div
                        className="rag-sources-header"
                        onClick={() => toggleSourceExpansion(msg.id)}
                      >
                        <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                          <Bookmark size={13} />
                          Retrieved Sources ({msg.sources.length})
                          {uniqueSourcesDocCount > 1 && (
                            <span
                              style={{
                                fontSize: '0.6875rem',
                                color: 'var(--primary-600)',
                                backgroundColor: 'var(--primary-50)',
                                padding: '1px 6px',
                                borderRadius: 'var(--radius-full)',
                                marginLeft: '4px',
                              }}
                            >
                              {uniqueSourcesDocCount} files
                            </span>
                          )}
                        </span>
                        {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      </div>

                      {isExpanded && (
                        <div className="rag-sources-list">
                          {msg.sources.map((src, sIdx) => {
                            const pageInfo = src.source_pages
                              ? `Page ${src.source_pages}`
                              : src.slide_number !== null && src.slide_number !== undefined && src.slide_number >= 0
                              ? `Slide ${src.slide_number}`
                              : `Chunk ${src.chunk_index + 1}`;
                            const matchPercent = Math.round((src.score || 0) * 100);

                            return (
                              <div key={sIdx} className="rag-source-card">
                                <div className="rag-source-meta">
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    <span className="rag-source-doc-badge">
                                      <FileText size={10} />
                                      {src.filename}
                                    </span>
                                    <span>• {pageInfo}</span>
                                  </div>
                                  <span className="rag-source-score">{matchPercent}% match</span>
                                </div>
                                <div className="rag-source-snippet">{src.content}</div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="message-meta">
                    <span>{msg.timestamp}</span>
                    {msg.model && <span>• {msg.model}</span>}
                  </div>
                </div>
              </div>
            );
          })
        )}

        {/* Loading / Thinking Indicator */}
        {isLoading && (
          <div className="message-row assistant">
            <div className="message-avatar assistant-avatar-icon">
              <Bot size={16} />
            </div>
            <div className="typing-bubble">
              <Sparkles size={14} style={{ color: 'var(--primary-500)' }} />
              <span>DocuMind is retrieving document context and synthesizing answer...</span>
              <div className="typing-dots">
                <div className="typing-dot" />
                <div className="typing-dot" />
                <div className="typing-dot" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <div className="chat-input-bar-container">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="chat-input-wrapper"
        >
          <textarea
            ref={textareaRef}
            className="chat-textarea-input"
            rows={1}
            placeholder={
              isAllDocsMode
                ? 'Ask a question across all documents (Enter to send, Shift+Enter for newline)...'
                : isSingleCustomSelected && activeSelectedDocs[0]
                ? `Ask a question about "${activeSelectedDocs[0].name}"...`
                : `Ask a question across ${activeSelectedDocs.length} selected documents...`
            }
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button
            type="submit"
            className="chat-send-btn"
            disabled={!inputPrompt.trim() || isLoading}
            aria-label="Send question"
          >
            <Send size={15} />
          </button>
        </form>
        <div className="chat-disclaimer">
          Answers are generated using Groq (GPT-OSS 120B) based strictly on indexed document chunks.
        </div>
      </div>
    </div>
  );
};

export default AIChatPage;
