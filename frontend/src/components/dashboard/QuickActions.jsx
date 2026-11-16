import React from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, MessageSquare, FileText, ArrowRight } from 'lucide-react';
import { useDocuments } from '../../context/DocumentContext';

const QuickActions = () => {
  const navigate = useNavigate();
  const { setIsUploadModalOpen } = useDocuments();

  const actions = [
    {
      id: 'upload',
      title: 'Upload Document',
      description: 'Ingest multi-format files (PDF, DOCX, TXT, CSV) into the RAG vector index.',
      icon: UploadCloud,
      onClick: () => setIsUploadModalOpen(true),
    },
    {
      id: 'chat',
      title: 'Chat with Documents',
      description: 'Ask questions with conversational AI and receive precise answers with citations.',
      icon: MessageSquare,
      onClick: () => navigate('/chat'),
    },
    {
      id: 'summarize',
      title: 'Generate Summary',
      description: 'Extract executive summaries or detailed section breakdowns with one click.',
      icon: FileText,
      onClick: () => navigate('/summarize'),
    },
  ];

  return (
    <div className="quick-actions-section">
      <div className="section-header">
        <h2 className="section-title">Quick Actions</h2>
        <span className="section-desc">Jump straight into document workflows</span>
      </div>

      <div className="quick-actions-grid">
        {actions.map((act) => {
          const Icon = act.icon;
          return (
            <div
              key={act.id}
              className="quick-action-card"
              onClick={act.onClick}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && act.onClick()}
            >
              <div className="quick-action-icon-box">
                <Icon size={22} />
              </div>
              <div className="quick-action-content">
                <div className="quick-action-title">
                  <span>{act.title}</span>
                  <ArrowRight size={16} className="quick-action-arrow" />
                </div>
                <p className="quick-action-description">{act.description}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default QuickActions;
