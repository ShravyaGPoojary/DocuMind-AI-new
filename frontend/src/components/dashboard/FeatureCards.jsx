import React from 'react';
import { Search, Brain, FileCheck, Layers } from 'lucide-react';

const FEATURES = [
  {
    icon: Search,
    title: 'Hybrid Semantic Retrieval',
    description: 'Dense vector embeddings paired with sparse BM25 indexing for high-precision document chunk discovery.',
    stage: 'Stage 2 Ingestion',
  },
  {
    icon: Brain,
    title: 'Contextual RAG Grounding',
    description: 'LLM responses anchored directly to verified document source paragraphs to eliminate hallucination.',
    stage: 'Stage 2 Retrieval',
  },
  {
    icon: FileCheck,
    title: 'Precision Citations',
    description: 'Inspect exact source page numbers, chunk offsets, and similarity metrics for every generated statement.',
    stage: 'Stage 2 Citations',
  },
  {
    icon: Layers,
    title: 'Multi-Document Synthesis',
    description: 'Cross-reference contracts, financial audits, and technical specs side-by-side with holistic answers.',
    stage: 'Stage 2 Synthesis',
  },
];

const FeatureCards = () => {
  return (
    <div className="quick-actions-section">
      <div className="section-header">
        <h2 className="section-title">Core Capabilities & RAG Architecture</h2>
        <span className="section-desc">Platform capabilities planned for Stage 2 RAG activation</span>
      </div>

      <div className="features-grid">
        {FEATURES.map((feat, idx) => {
          const Icon = feat.icon;
          return (
            <div key={idx} className="feature-overview-card">
              <div className="feature-icon-wrapper">
                <Icon size={20} />
              </div>
              <h3 className="feature-title">{feat.title}</h3>
              <p className="feature-description">{feat.description}</p>
              <span className="feature-badge-stage">{feat.stage}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default FeatureCards;
