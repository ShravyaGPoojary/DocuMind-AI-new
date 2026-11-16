import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  Search,
  LayoutGrid,
  List,
  Eye,
  Trash2,
  FileText,
  Plus,
} from 'lucide-react';
import Button from '../components/common/Button';
import Badge, { FileTypeBadge } from '../components/common/Badge';
import EmptyState from '../components/common/EmptyState';
import { useDocuments } from '../context/DocumentContext';

const DocumentsPage = () => {
  const {
    documents,
    searchQuery,
    setSearchQuery,
    setSelectedDocForDetails,
    deleteDocument,
    addDocument,
    setIsUploadModalOpen,
  } = useDocuments();

  const [viewMode, setViewMode] = useState('grid'); // 'grid' | 'table'
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = () => {
    setIsDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      addDocument(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      addDocument(e.target.files[0]);
    }
  };

  // Filtered documents by search query
  const filteredDocuments = documents.filter((doc) =>
    doc.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="documents-page">
      {/* Header Row */}
      <div className="page-header-row">
        <div className="page-header-title-group">
          <h1>Document Library</h1>
          <p>
            Upload, manage, and inspect documents for AI analysis.
          </p>
        </div>

        <Button
          variant="primary"
          icon={Plus}
          onClick={() => setIsUploadModalOpen(true)}
        >
          Upload Document
        </Button>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        className={`upload-dropzone ${isDragActive ? 'drag-active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept=".pdf,.docx,.pptx,.txt,.md,.csv"
          className="hidden-file-input"
        />
        <div className="upload-icon-circle">
          <UploadCloud size={24} />
        </div>
        <div className="upload-title">Drop your documents here, or click to browse</div>
        <div className="upload-subtitle">
          Supports PDF, Word (.docx), PowerPoint (.pptx), Plain Text (.txt), Markdown (.md), and CSV
        </div>
      </div>

      {/* Toolbar: Search & View Switcher */}
      <div className="documents-toolbar">
        <div className="toolbar-left">
          <div className="search-input-wrapper">
            <Search size={15} className="search-input-icon" />
            <input
              type="text"
              className="filter-input"
              placeholder="Search documents by name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        <div className="toolbar-right">
          <span style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>
            {filteredDocuments.length} document{filteredDocuments.length !== 1 ? 's' : ''}
          </span>

          <div className="view-toggle-group">
            <button
              className={`view-toggle-btn ${viewMode === 'grid' ? 'active' : ''}`}
              onClick={() => setViewMode('grid')}
              title="Grid View"
              aria-label="Grid View"
            >
              <LayoutGrid size={16} />
            </button>
            <button
              className={`view-toggle-btn ${viewMode === 'table' ? 'active' : ''}`}
              onClick={() => setViewMode('table')}
              title="Table View"
              aria-label="Table View"
            >
              <List size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Document Content: Grid / Table / Empty State */}
      {filteredDocuments.length === 0 ? (
        <EmptyState
          icon={FileText}
          title={
            documents.length === 0
              ? 'No documents in your library'
              : 'No matching documents found'
          }
          description={
            documents.length === 0
              ? 'Your document library is empty. Upload your first document using the upload area above.'
              : `No documents match "${searchQuery}".`
          }
          actionText={documents.length === 0 ? 'Upload Document' : 'Clear Search'}
          actionIcon={documents.length === 0 ? Plus : undefined}
          onAction={() => {
            if (documents.length === 0) {
              setIsUploadModalOpen(true);
            } else {
              setSearchQuery('');
            }
          }}
        />
      ) : viewMode === 'grid' ? (
        <div className="documents-grid">
          {filteredDocuments.map((doc) => (
            <div key={doc.id} className="doc-card">
              <div className="doc-card-header">
                <div className="doc-icon-title">
                  <FileTypeBadge type={doc.file_type} />
                  <div>
                    <h3 className="doc-card-title">{doc.name}</h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
                      Uploaded on {doc.uploaded_at}
                    </span>
                  </div>
                </div>
                <Badge variant={doc.status}>{doc.status.toUpperCase()}</Badge>
              </div>

              <div className="doc-card-meta-list">
                <span>Format: {doc.file_type}</span>
                <span>•</span>
                <span>Size: {doc.size_formatted}</span>
              </div>

              <div className="doc-card-actions">
                <Button
                  variant="secondary"
                  size="sm"
                  icon={Eye}
                  onClick={() => setSelectedDocForDetails(doc)}
                >
                  Details
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  icon={Trash2}
                  onClick={() => deleteDocument(doc.id)}
                  title="Remove document"
                >
                  Delete
                </Button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* Table View */
        <div
          style={{
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-lg)',
            overflow: 'hidden',
          }}
        >
          <table className="data-table">
            <thead>
              <tr>
                <th>Document</th>
                <th>Format</th>
                <th>Size</th>
                <th>Uploaded</th>
                <th>Status</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredDocuments.map((doc) => (
                <tr key={doc.id}>
                  <td>
                    <div className="doc-name-cell">
                      <FileTypeBadge type={doc.file_type} />
                      <span style={{ fontWeight: 500 }}>{doc.name}</span>
                    </div>
                  </td>
                  <td>
                    <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>{doc.file_type}</span>
                  </td>
                  <td>
                    <span style={{ color: 'var(--text-secondary)' }}>{doc.size_formatted}</span>
                  </td>
                  <td>
                    <span style={{ color: 'var(--text-secondary)' }}>{doc.uploaded_at}</span>
                  </td>
                  <td>
                    <Badge variant={doc.status}>{doc.status.toUpperCase()}</Badge>
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <div style={{ display: 'inline-flex', gap: '6px' }}>
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={Eye}
                        onClick={() => setSelectedDocForDetails(doc)}
                      >
                        Inspect
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={Trash2}
                        onClick={() => deleteDocument(doc.id)}
                      >
                        Delete
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default DocumentsPage;
