import React from 'react';
import { Link } from 'react-router-dom';
import { FileText, ArrowRight, Eye, Plus } from 'lucide-react';
import Card, { CardHeader, CardBody } from '../common/Card';
import Badge, { FileTypeBadge } from '../common/Badge';
import Button from '../common/Button';
import EmptyState from '../common/EmptyState';
import { useDocuments } from '../../context/DocumentContext';

const RecentDocuments = () => {
  const { documents, setSelectedDocForDetails, setIsUploadModalOpen } = useDocuments();

  const recentDocs = documents.slice(0, 5);

  return (
    <Card>
      <CardHeader
        title="Recent Documents"
        icon={FileText}
        action={
          documents.length > 0 ? (
            <Link
              to="/documents"
              style={{
                fontSize: '0.8125rem',
                color: 'var(--primary-600)',
                fontWeight: 500,
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <span>View All ({documents.length})</span>
              <ArrowRight size={14} />
            </Link>
          ) : null
        }
      />
      <CardBody style={{ padding: 0 }}>
        {recentDocs.length === 0 ? (
          <div style={{ padding: '32px 24px' }}>
            <EmptyState
              title="No documents yet"
              description="Your document library is currently empty. Upload a PDF, Word, PowerPoint, plain text, markdown, or CSV document to get started."
              actionText="Upload Document"
              actionIcon={Plus}
              onAction={() => setIsUploadModalOpen(true)}
            />
          </div>
        ) : (
          <div className="recent-docs-table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Document Name</th>
                  <th>Format</th>
                  <th>Size</th>
                  <th>Uploaded</th>
                  <th>Status</th>
                  <th style={{ textAlign: 'right' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {recentDocs.map((doc) => (
                  <tr key={doc.id}>
                    <td>
                      <div className="doc-name-cell">
                        <FileTypeBadge type={doc.file_type} />
                        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                          {doc.name}
                        </span>
                      </div>
                    </td>
                    <td>
                      <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                        {doc.file_type}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                        {doc.size_formatted}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                        {doc.uploaded_at}
                      </span>
                    </td>
                    <td>
                      <Badge variant={doc.status}>
                        {doc.status.toUpperCase()}
                      </Badge>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={Eye}
                        onClick={() => setSelectedDocForDetails(doc)}
                      >
                        Inspect
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardBody>
    </Card>
  );
};

export default RecentDocuments;
