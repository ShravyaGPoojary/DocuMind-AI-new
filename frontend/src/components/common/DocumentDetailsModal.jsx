import React, { useState, useEffect } from 'react';
import { CheckCircle2, Layers, BookOpen, AlertCircle, Loader2 } from 'lucide-react';
import Modal from './Modal';
import Button from './Button';
import Badge, { FileTypeBadge } from './Badge';
import { getDocumentChunks } from '../../services/api';

const DocumentDetailsModal = ({ document, isOpen, onClose }) => {
  const [chunks, setChunks] = useState([]);
  const [isLoadingChunks, setIsLoadingChunks] = useState(false);
  const [chunksError, setChunksError] = useState(null);

  useEffect(() => {
    if (isOpen && document?.id) {
      let isMounted = true;
      setIsLoadingChunks(true);
      setChunksError(null);

      getDocumentChunks(document.id, 1, 10)
        .then((data) => {
          if (isMounted) {
            setChunks(data.chunks || []);
          }
        })
        .catch((err) => {
          if (isMounted) {
            setChunksError(err.message || 'Failed to load chunks preview.');
          }
        })
        .finally(() => {
          if (isMounted) {
            setIsLoadingChunks(false);
          }
        });

      return () => {
        isMounted = false;
      };
    } else {
      setChunks([]);
    }
  }, [isOpen, document?.id]);

  if (!document) return null;

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Document Details & Chunks"
      footer={
        <Button variant="secondary" onClick={onClose}>
          Close Details
        </Button>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <FileTypeBadge type={document.file_type} />
          <div>
            <h4 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {document.name}
            </h4>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
              ID: {document.id}
            </span>
          </div>
        </div>

        <div className="doc-detail-grid">
          <div className="detail-item">
            <span className="detail-label">File Format</span>
            <span className="detail-value">{document.file_type}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">File Size</span>
            <span className="detail-value">{document.size_formatted}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Upload Date</span>
            <span className="detail-value">{document.uploaded_at}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Status</span>
            <span className="detail-value">
              <Badge variant={document.status}>
                <CheckCircle2 size={12} />
                {document.status.toUpperCase()}
              </Badge>
            </span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Total Chunks</span>
            <span className="detail-value" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Layers size={13} />
              {document.chunks_count || chunks.length || 0}
            </span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Pages / Slides</span>
            <span className="detail-value" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <BookOpen size={13} />
              {document.pages ? `${document.pages}` : 'N/A'}
            </span>
          </div>
        </div>

        {/* Chunks Preview Section */}
        <div style={{ marginTop: '8px' }}>
          <div
            style={{
              fontSize: '0.875rem',
              fontWeight: 600,
              color: 'var(--text-primary)',
              marginBottom: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <span>Chunks Preview (first 10)</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontWeight: 400 }}>
              500 words target • 100 overlap
            </span>
          </div>

          {isLoadingChunks ? (
            <div
              style={{
                padding: '24px',
                textAlign: 'center',
                color: 'var(--text-tertiary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                fontSize: '0.8125rem',
              }}
            >
              <Loader2 size={16} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
              Loading chunks...
            </div>
          ) : chunksError ? (
            <div
              style={{
                padding: '12px',
                backgroundColor: 'var(--danger-bg, #fee2e2)',
                color: 'var(--danger-text, #b91c1c)',
                borderRadius: 'var(--radius-md)',
                fontSize: '0.8125rem',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <AlertCircle size={15} />
              <span>{chunksError}</span>
            </div>
          ) : chunks.length === 0 ? (
            <div
              style={{
                padding: '16px',
                textAlign: 'center',
                color: 'var(--text-tertiary)',
                fontSize: '0.8125rem',
                border: '1px dashed var(--border-color)',
                borderRadius: 'var(--radius-md)',
              }}
            >
              No chunks generated for this document yet.
            </div>
          ) : (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                maxHeight: '260px',
                overflowY: 'auto',
                paddingRight: '4px',
              }}
            >
              {chunks.map((chunk) => (
                <div
                  key={chunk.id}
                  style={{
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-md)',
                    padding: '10px 12px',
                    fontSize: '0.8125rem',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      marginBottom: '6px',
                      fontSize: '0.75rem',
                      color: 'var(--text-tertiary)',
                    }}
                  >
                    <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
                      Chunk #{chunk.chunk_index + 1}
                    </span>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      {chunk.source_pages && (
                        <span>Page {chunk.source_pages}</span>
                      )}
                      {chunk.slide_number && (
                        <span>Slide {chunk.slide_number}</span>
                      )}
                      <span>{chunk.word_count} words</span>
                    </div>
                  </div>
                  <p
                    style={{
                      margin: 0,
                      color: 'var(--text-primary)',
                      lineHeight: 1.45,
                      display: '-webkit-box',
                      WebkitLineClamp: 3,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                      fontSize: '0.75rem',
                    }}
                  >
                    {chunk.content}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
};

export default DocumentDetailsModal;
