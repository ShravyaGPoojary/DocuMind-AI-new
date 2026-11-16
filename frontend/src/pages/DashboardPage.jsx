import React from 'react';
import { FileText, CheckCircle2, Plus } from 'lucide-react';
import StatCard from '../components/common/StatCard';
import Button from '../components/common/Button';
import RecentDocuments from '../components/dashboard/RecentDocuments';
import { useDocuments } from '../context/DocumentContext';

const DashboardPage = () => {
  const { documents, setIsUploadModalOpen } = useDocuments();

  const totalDocs = documents.length;
  const processedDocs = documents.filter((d) => d.status === 'ready').length;

  return (
    <div className="dashboard-page">
      {/* Welcome Header */}
      <div className="dashboard-hero">
        <div className="hero-text-content">
          <div className="hero-greeting">
            <h1 className="hero-title">Welcome to DocuMind AI</h1>
          </div>
          <p className="hero-subtitle">
            An Intelligent Multi-Format Document Understanding Platform powered by Retrieval-Augmented Generation (RAG).
            Upload documents to chat, generate summaries, and perform multi-document synthesis with verified source citations.
          </p>
        </div>

        <div className="hero-actions">
          <Button
            variant="primary"
            size="md"
            icon={Plus}
            onClick={() => setIsUploadModalOpen(true)}
          >
            Upload Document
          </Button>
        </div>
      </div>

      {/* Statistics Row (2 clean metric cards) */}
      <div className="stats-grid simplified">
        <StatCard
          label="Total Documents"
          value={totalDocs}
          subtext="Documents in library"
          icon={FileText}
        />
        <StatCard
          label="Processed Documents"
          value={processedDocs}
          subtext="Ready for RAG analysis"
          icon={CheckCircle2}
        />
      </div>

      {/* Recent Documents Table & Empty State */}
      <RecentDocuments />
    </div>
  );
};

export default DashboardPage;
