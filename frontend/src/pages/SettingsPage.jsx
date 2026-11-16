import React from 'react';
import {
  Server,
  FileCheck,
  Info,
  Layers,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import Button from '../components/common/Button';
import Badge from '../components/common/Badge';
import { useDocuments } from '../context/DocumentContext';

const SUPPORTED_FORMATS_DATA = [
  { ext: 'PDF', name: 'Portable Document Format', limit: '25 MB', desc: 'Reports, research papers, documentation' },
  { ext: 'DOCX', name: 'Microsoft Word Document', limit: '25 MB', desc: 'Specifications, proposals, manuals' },
  { ext: 'TXT', name: 'Plain Text File', limit: '10 MB', desc: 'Notes, transcripts, system logs' },
  { ext: 'MD', name: 'Markdown Document', limit: '10 MB', desc: 'READMEs, runbooks, documentation' },
  { ext: 'CSV', name: 'Comma-Separated Values', limit: '20 MB', desc: 'Tabular data, spreadsheets' },
];

const SettingsPage = () => {
  const { backendStatus, backendInfo, refreshData } = useDocuments();

  return (
    <div className="settings-page">
      <div className="page-header-row">
        <div className="page-header-title-group">
          <h1>Settings & System Information</h1>
          <p>
            System details, backend connection status, and supported document formats.
          </p>
        </div>
      </div>

      {/* Project Information Section */}
      <div className="settings-section-card">
        <div className="settings-section-header">
          <div className="settings-section-title-wrap">
            <h2 className="settings-section-title">
              <Info size={18} color="var(--primary-600)" />
              Project Information
            </h2>
            <p className="settings-section-subtitle">
              Application specifications and runtime architecture
            </p>
          </div>
        </div>

        <div className="settings-row">
          <div className="settings-row-label-group">
            <span className="settings-row-title">Application Name</span>
            <span className="settings-row-desc">DocuMind AI</span>
          </div>
          <div className="settings-control" style={{ fontSize: '0.875rem', fontWeight: 500 }}>
            DocuMind AI
          </div>
        </div>

        <div className="settings-row">
          <div className="settings-row-label-group">
            <span className="settings-row-title">Version</span>
            <span className="settings-row-desc">Current release version tag</span>
          </div>
          <div className="settings-control">
            <Badge variant="primary">v1.0.0</Badge>
          </div>
        </div>

        <div className="settings-row">
          <div className="settings-row-label-group">
            <span className="settings-row-title">Architecture</span>
            <span className="settings-row-desc">Frontend and backend stack</span>
          </div>
          <div className="settings-control" style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
            React 18 + Vite + FastAPI (Python)
          </div>
        </div>

        <div className="settings-row">
          <div className="settings-row-label-group">
            <span className="settings-row-title">Current Stage</span>
            <span className="settings-row-desc">Development phase status</span>
          </div>
          <div className="settings-control">
            <Badge variant="ready">Stage 1: Complete UI + Backend Foundation</Badge>
          </div>
        </div>
      </div>

      {/* Backend API Connection Status */}
      <div className="settings-section-card">
        <div className="settings-section-header">
          <div className="settings-section-title-wrap">
            <h2 className="settings-section-title">
              <Server size={18} color="var(--primary-600)" />
              Backend Connection Status
            </h2>
            <p className="settings-section-subtitle">
              Live status pinging FastAPI backend at http://localhost:8000/api/health
            </p>
          </div>

          <Button variant="secondary" size="sm" icon={RefreshCw} onClick={refreshData}>
            Ping Backend
          </Button>
        </div>

        <div className="settings-row">
          <div className="settings-row-label-group">
            <span className="settings-row-title">Connection Status</span>
            <span className="settings-row-desc">
              Connectivity state between React frontend and FastAPI server
            </span>
          </div>
          <div className="settings-control">
            <Badge variant={backendStatus === 'online' ? 'ready' : 'error'}>
              {backendStatus === 'online' ? (
                <>
                  <CheckCircle2 size={13} />
                  FASTAPI ONLINE
                </>
              ) : (
                <>
                  <AlertCircle size={13} />
                  BACKEND OFFLINE
                </>
              )}
            </Badge>
          </div>
        </div>

        <div className="settings-row">
          <div className="settings-row-label-group">
            <span className="settings-row-title">API Endpoint</span>
            <span className="settings-row-desc">Base URL used for API requests</span>
          </div>
          <div className="settings-control" style={{ fontSize: '0.8125rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
            http://localhost:8000/api
          </div>
        </div>
      </div>

      {/* Supported Document Formats */}
      <div className="settings-section-card">
        <div className="settings-section-header">
          <div className="settings-section-title-wrap">
            <h2 className="settings-section-title">
              <FileCheck size={18} color="var(--primary-600)" />
              Supported Document Formats
            </h2>
            <p className="settings-section-subtitle">
              Accepted file types and maximum upload limits
            </p>
          </div>
        </div>

        <div className="formats-matrix-grid">
          {SUPPORTED_FORMATS_DATA.map((fmt) => (
            <div key={fmt.ext} className="format-matrix-card">
              <div className="format-matrix-icon">.{fmt.ext}</div>
              <div className="format-matrix-info">
                <span className="format-matrix-name">{fmt.name}</span>
                <span className="format-matrix-desc">Max Size: {fmt.limit}</span>
                <span className="format-matrix-desc">{fmt.desc}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
