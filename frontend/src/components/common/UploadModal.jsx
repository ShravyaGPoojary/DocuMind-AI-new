import React, { useState, useRef } from 'react';
import { UploadCloud, File, AlertTriangle, CheckCircle2, Loader2, XCircle } from 'lucide-react';
import Modal from './Modal';
import Button from './Button';
import { useDocuments } from '../../context/DocumentContext';

const UploadModal = ({ isOpen, onClose }) => {
  const { addDocument, isUploading } = useDocuments();
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    if (isUploading) return;
    setIsDragActive(true);
  };

  const handleDragLeave = () => {
    setIsDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (isUploading) return;
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
      setUploadError(null);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setUploadError(null);
    }
  };

  const handleConfirmUpload = async () => {
    if (!selectedFile || isUploading) return;
    setUploadError(null);
    try {
      await addDocument(selectedFile);
      handleReset();
      onClose();
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        'Upload failed. Please check the file format and size.';
      setUploadError(msg);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setUploadError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={() => {
        if (isUploading) return;
        handleReset();
        onClose();
      }}
      title="Upload Document"
      footer={
        <>
          <Button
            variant="secondary"
            disabled={isUploading}
            onClick={() => {
              handleReset();
              onClose();
            }}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            disabled={!selectedFile || isUploading}
            onClick={handleConfirmUpload}
            icon={isUploading ? Loader2 : UploadCloud}
          >
            {isUploading ? 'Processing Document...' : 'Upload & Process'}
          </Button>
        </>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div
          className={`upload-dropzone ${isDragActive ? 'drag-active' : ''} ${isUploading ? 'uploading-active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => {
            if (!isUploading) fileInputRef.current?.click();
          }}
          style={{
            padding: '32px 20px',
            cursor: isUploading ? 'not-allowed' : 'pointer',
            opacity: isUploading ? 0.7 : 1,
          }}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf,.docx,.pptx,.txt,.md,.csv"
            disabled={isUploading}
            style={{ display: 'none' }}
          />
          <div className="upload-icon-circle">
            {isUploading ? (
              <Loader2 size={24} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
            ) : (
              <UploadCloud size={24} />
            )}
          </div>
          <div className="upload-title">
            {isUploading
              ? 'Extracting text and chunking document...'
              : selectedFile
              ? selectedFile.name
              : 'Click to select or drag and drop file'}
          </div>
          <div className="upload-subtitle">
            {isUploading
              ? 'Running text extraction, normalization, and 500-word sliding window chunking.'
              : selectedFile
              ? `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for processing`
              : 'Supported: PDF, DOCX, PPTX, TXT, MD, CSV (Max 25MB)'}
          </div>

          <div className="supported-formats-pills">
            <span className="format-chip">.PDF</span>
            <span className="format-chip">.DOCX</span>
            <span className="format-chip">.PPTX</span>
            <span className="format-chip">.TXT</span>
            <span className="format-chip">.MD</span>
            <span className="format-chip">.CSV</span>
          </div>
        </div>

        {uploadError && (
          <div
            style={{
              padding: '12px',
              backgroundColor: 'var(--danger-bg, #fee2e2)',
              border: '1px solid var(--danger-border, #fca5a5)',
              borderRadius: 'var(--radius-md)',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              fontSize: '0.8125rem',
              color: 'var(--danger-text, #b91c1c)',
            }}
          >
            <XCircle size={16} style={{ flexShrink: 0 }} />
            <span>{uploadError}</span>
          </div>
        )}

        {selectedFile && !uploadError && (
          <div
            style={{
              padding: '12px',
              backgroundColor: 'var(--success-bg)',
              border: '1px solid var(--success-border)',
              borderRadius: 'var(--radius-md)',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              fontSize: '0.8125rem',
              color: 'var(--success-text)',
            }}
          >
            <CheckCircle2 size={16} />
            <span>
              File ready: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024).toFixed(1)} KB)
            </span>
          </div>
        )}

        {/* Stage 2 Information Box */}
        <div
          style={{
            padding: '12px',
            backgroundColor: 'var(--info-bg)',
            border: '1px solid var(--info-border)',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '10px',
            fontSize: '0.75rem',
            color: 'var(--info-text)',
            lineHeight: 1.45,
          }}
        >
          <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong>Stage 2 Pipeline Active:</strong> Uploading triggers real server-side text extraction, text cleaning, 500-word sliding window chunking with 100-word overlap, and persistent storage in SQLite.
          </div>
        </div>
      </div>
    </Modal>
  );
};

export default UploadModal;
