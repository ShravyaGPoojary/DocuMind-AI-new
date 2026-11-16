import React, { useState } from 'react';
import {
  Boxes,
  Sparkles,
  Bookmark,
  CheckSquare,
  Square,
  FileText,
  Plus,
  AlertCircle,
  Clock,
  Layers,
} from 'lucide-react';
import Button from '../components/common/Button';
import Card, { CardHeader, CardBody } from '../components/common/Card';
import EmptyState from '../components/common/EmptyState';
import { useDocuments } from '../context/DocumentContext';
import { sendChatMessage } from '../services/api';

const EXAMPLE_MULTI_QUESTIONS = [
  'What are the common themes or overlapping objectives across the selected documents?',
  'Compare the key recommendations, methodology, or conclusions between these files.',
  'Identify any conflicting points, differences in metrics, or distinct focuses across these documents.',
];

const MultiDocQAPage = () => {
  const { documents, selectedDocIds, toggleDocSelection, setSelectedDocIds, setIsUploadModalOpen } =
    useDocuments();
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [synthesisResult, setSynthesisResult] = useState(null);

  const selectedDocs = documents.filter((d) => selectedDocIds.includes(d.id));

  const handleSelectAll = () => {
    setSelectedDocIds(documents.map((d) => d.id));
  };

  const handleDeselectAll = () => {
    setSelectedDocIds([]);
  };

  const handleAskMultiDoc = async (sampleQuery) => {
    const activeQuery = (sampleQuery || query).trim();
    if (!activeQuery || selectedDocs.length === 0 || isProcessing) return;

    setIsProcessing(true);
    setErrorMessage(null);

    try {
      const targetDocIds = selectedDocs.map((d) => d.id);
      const topK = Math.max(4, Math.min(15, targetDocIds.length * 3));

      const response = await sendChatMessage({
        question: activeQuery,
        documentIds: targetDocIds,
        topK,
        timeout: 180000,
      });

      setSynthesisResult({
        query: activeQuery,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        targetDocs: selectedDocs.map((d) => d.name),
        answer: response.answer,
        model: response.model || 'Groq GPT-OSS 120B',
        citations: response.sources || [],
      });
    } catch (err) {
      console.error('Multi-Doc QA failed:', err);
      let detail =
        err.response?.data?.detail ||
        err.message ||
        'Failed to generate multi-document answer.';

      if (err.response?.status === 429) {
        detail = 'Groq rate limit reached. Please wait a few seconds before asking another question.';
      } else if (err.response?.status === 504 || err.code === 'ECONNABORTED') {
        detail = 'Multi-document synthesis timed out. Try asking a more specific question.';
      }
      setErrorMessage(detail);
    } finally {
      setIsProcessing(false);
    }
  };

  if (documents.length === 0) {
    return (
      <div className="multidoc-page">
        <div className="page-header-row">
          <div className="page-header-title-group">
            <h1>Multi-Document Q&A</h1>
            <p>Select multiple documents to synthesize comparative answers with source citations.</p>
          </div>
        </div>
        <EmptyState
          icon={Boxes}
          title="No documents uploaded yet"
          description="Please upload at least two documents to perform multi-document comparative synthesis."
          actionText="Upload Document"
          actionIcon={Plus}
          onAction={() => setIsUploadModalOpen(true)}
        />
      </div>
    );
  }

  return (
    <div className="multidoc-page">
      <div className="page-header-row">
        <div className="page-header-title-group">
          <h1>Multi-Document Q&A</h1>
          <p>
            Select 2 or more documents to correlate insights and compare information.
          </p>
        </div>
      </div>

      <div className="multidoc-layout">
        {/* Left Document Checklist */}
        <Card className="doc-selection-panel">
          <CardHeader
            title="Select Documents"
            icon={Boxes}
            action={
              <span className="selection-count-badge">
                {selectedDocs.length} of {documents.length} Selected
              </span>
            }
          />
          <CardBody style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
              <button
                type="button"
                onClick={handleSelectAll}
                style={{ background: 'none', border: 'none', color: 'var(--primary-600)', cursor: 'pointer', fontWeight: 500 }}
              >
                Select All
              </button>
              <button
                type="button"
                onClick={handleDeselectAll}
                style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer' }}
              >
                Deselect All
              </button>
            </div>

            <div className="doc-checklist">
              {documents.map((doc) => {
                const isChecked = selectedDocIds.includes(doc.id);
                return (
                  <div
                    key={doc.id}
                    className={`doc-check-item ${isChecked ? 'checked' : ''}`}
                    onClick={() => toggleDocSelection(doc.id)}
                  >
                    <div className="custom-checkbox">
                      {isChecked ? <CheckSquare size={16} /> : <Square size={16} color="var(--border-strong)" />}
                    </div>
                    <div className="doc-check-info">
                      <span className="doc-check-name truncate" title={doc.name}>
                        {doc.name}
                      </span>
                      <span className="doc-check-meta">
                        {doc.file_type} • {doc.size_formatted}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardBody>
        </Card>

        {/* Right Query & Synthesis Workspace */}
        <div className="multidoc-workspace">
          {/* Query Box Card */}
          <div className="query-card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <label style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                Ask a Cross-Document Question
              </label>
              <div className="selected-docs-pill-strip">
                {selectedDocs.slice(0, 2).map((d) => (
                  <span key={d.id} className="active-doc-tag truncate" style={{ maxWidth: '140px' }}>
                    {d.name}
                  </span>
                ))}
                {selectedDocs.length > 2 && (
                  <span className="active-doc-tag">+{selectedDocs.length - 2} more</span>
                )}
              </div>
            </div>

            <textarea
              className="query-textarea"
              placeholder={
                selectedDocs.length === 0
                  ? 'Select at least one document from the list on the left...'
                  : `Ask a question across the ${selectedDocs.length} selected document(s)...`
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={selectedDocs.length === 0}
            />

            <div className="query-actions-row">
              <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
                {selectedDocs.length} document{selectedDocs.length !== 1 ? 's' : ''} selected
              </span>

              <Button
                variant="primary"
                icon={Sparkles}
                loading={isProcessing}
                disabled={!query.trim() || selectedDocs.length === 0 || isProcessing}
                onClick={() => handleAskMultiDoc()}
              >
                Ask Multi-Doc AI
              </Button>
            </div>
          </div>

          {/* Suggested Questions */}
          <div className="multi-examples-row">
            <span className="examples-heading">Example questions:</span>
            <div className="example-chips-wrap">
              {EXAMPLE_MULTI_QUESTIONS.map((eq, idx) => (
                <div
                  key={idx}
                  className="example-chip"
                  onClick={() => {
                    setQuery(eq);
                    if (selectedDocs.length > 0) handleAskMultiDoc(eq);
                  }}
                >
                  {eq}
                </div>
              ))}
            </div>
          </div>

          {/* Error Banner */}
          {errorMessage && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '10px 14px',
                backgroundColor: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: 'var(--radius-md)',
                color: '#991b1b',
                fontSize: '0.8125rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <AlertCircle size={16} />
                <span>{errorMessage}</span>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setErrorMessage(null)}>
                Dismiss
              </Button>
            </div>
          )}

          {/* Synthesis Answer Box */}
          {synthesisResult ? (
            <div className="synthesis-card">
              <div className="synthesis-header">
                <div className="synthesis-title">
                  <Sparkles size={18} color="var(--primary-600)" />
                  <span>Synthesized Multi-Document Answer</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
                  {synthesisResult.model && (
                    <span
                      style={{
                        backgroundColor: 'var(--primary-50)',
                        color: 'var(--primary-700)',
                        padding: '2px 8px',
                        borderRadius: 'var(--radius-full)',
                        fontWeight: 500,
                      }}
                    >
                      {synthesisResult.model}
                    </span>
                  )}
                  <span>{synthesisResult.timestamp}</span>
                </div>
              </div>

              <div className="synthesis-answer" style={{ whiteSpace: 'pre-wrap' }}>
                {synthesisResult.answer}
              </div>

              {synthesisResult.citations && synthesisResult.citations.length > 0 && (
                <div>
                  <span
                    style={{
                      fontSize: '0.8125rem',
                      fontWeight: 600,
                      color: 'var(--text-primary)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      marginBottom: '8px',
                    }}
                  >
                    <Bookmark size={14} color="var(--primary-600)" />
                    Grounded Source Citations ({synthesisResult.citations.length})
                  </span>

                  <div className="citations-grid">
                    {synthesisResult.citations.map((cite, cIdx) => {
                      const pageInfo = cite.source_pages
                        ? `Page ${cite.source_pages}`
                        : cite.slide_number !== null && cite.slide_number !== undefined && cite.slide_number >= 0
                        ? `Slide ${cite.slide_number}`
                        : `Chunk ${(cite.chunk_index !== undefined ? cite.chunk_index + 1 : cIdx + 1)}`;
                      const matchPercent = Math.round((cite.score || 0) * 100);

                      return (
                        <div key={cIdx} className="cross-citation-card">
                          <div className="cross-citation-source">
                            <FileText size={12} />
                            <span className="truncate" title={cite.filename}>
                              {cite.filename}
                            </span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6875rem', color: 'var(--text-tertiary)' }}>
                            <span>{pageInfo}</span>
                            {matchPercent > 0 && <span>{matchPercent}% match</span>}
                          </div>
                          {cite.content && (
                            <div
                              className="cross-citation-quote"
                              style={{
                                maxHeight: '90px',
                                overflowY: 'auto',
                                marginTop: '4px',
                                fontSize: '0.75rem',
                              }}
                            >
                              "{cite.content.slice(0, 220)}{cite.content.length > 220 ? '...' : ''}"
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <EmptyState
              icon={Boxes}
              title="No query submitted yet"
              description="Select documents and enter a prompt above to generate a comparative analysis."
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default MultiDocQAPage;
