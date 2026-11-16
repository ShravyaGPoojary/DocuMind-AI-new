import React, { useState, useEffect } from 'react';
import {
  GitCompare,
  Sparkles,
  ArrowRightLeft,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Copy,
  Check,
  Download,
  Info,
  Clock,
  Layers,
  ChevronDown,
  ExternalLink,
} from 'lucide-react';
import { useDocuments } from '../context/DocumentContext';
import { compareDocuments } from '../services/api';
import '../styles/comparison.css';

const TABS = [
  { id: 'summary', label: 'Executive Summary' },
  { id: 'findings', label: 'Similarities & Differences' },
  { id: 'concepts', label: 'Concepts & Themes' },
  { id: 'contradictions', label: 'Contradictions & Disagreements' },
  { id: 'agreements', label: 'Agreements' },
  { id: 'citations', label: 'Source Citations' },
];

const DocumentComparisonPage = () => {
  const { documents, fetchDocuments } = useDocuments();

  const [docId1, setDocId1] = useState('');
  const [docId2, setDocId2] = useState('');
  const [focusArea, setFocusArea] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [activeTab, setActiveTab] = useState('summary');
  const [copiedType, setCopiedType] = useState(null);

  // Load documents if not already loaded
  useEffect(() => {
    if (!documents || documents.length === 0) {
      fetchDocuments();
    }
  }, [documents, fetchDocuments]);

  // Pre-select first two ready documents if available
  useEffect(() => {
    const readyDocs = (documents || []).filter(
      (d) => d.status === 'ready' || d.processing_status === 'ready' || (d.chunk_count && d.chunk_count > 0)
    );
    if (readyDocs.length >= 2 && !docId1 && !docId2) {
      setDocId1(readyDocs[0].id);
      setDocId2(readyDocs[1].id);
    }
  }, [documents, docId1, docId2]);

  // Available ready documents
  const availableDocs = (documents || []).filter(
    (d) => d.status === 'ready' || d.processing_status === 'ready' || (d.chunk_count && d.chunk_count > 0)
  );

  const selectedDoc1 = availableDocs.find((d) => d.id === docId1);
  const selectedDoc2 = availableDocs.find((d) => d.id === docId2);

  const canCompare = docId1 && docId2 && docId1 !== docId2 && !isLoading;

  const handleCompare = async () => {
    if (!canCompare) return;

    setIsLoading(true);
    setError(null);
    setComparisonResult(null);

    try {
      const data = await compareDocuments(docId1, docId2, focusArea, 'groq');
      setComparisonResult(data);
      setActiveTab('summary');
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        'Failed to perform document comparison. Please ensure Groq API key is configured.';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopyMarkdown = () => {
    if (!comparisonResult) return;

    let md = `# Cross-Document Comparison Analysis\n\n`;
    md += `**Document 1:** ${comparisonResult.document_1.original_filename} (${comparisonResult.document_1.file_type})\n`;
    md += `**Document 2:** ${comparisonResult.document_2.original_filename} (${comparisonResult.document_2.file_type})\n`;
    if (comparisonResult.focus_area) {
      md += `**Focus Area:** ${comparisonResult.focus_area}\n`;
    }
    md += `**Model:** ${comparisonResult.model} | **Latency:** ${(comparisonResult.processing_time_ms / 1000).toFixed(2)}s\n\n`;

    md += `## 1. Executive Summary\n${comparisonResult.comparison_summary}\n\n`;

    md += `## 2. Similarities\n`;
    (comparisonResult.similarities || []).forEach((item, idx) => {
      md += `### ${idx + 1}. ${item.topic}\n${item.description}\n`;
      if (item.excerpt) md += `> ${item.excerpt}\n`;
      md += `\n`;
    });

    md += `## 3. Differences\n`;
    (comparisonResult.differences || []).forEach((item, idx) => {
      md += `### ${idx + 1}. ${item.topic} (${item.source_document || 'General'})\n${item.description}\n`;
      if (item.excerpt) md += `> ${item.excerpt}\n`;
      md += `\n`;
    });

    md += `## 4. Key Concepts\n`;
    md += `**Shared Concepts:** ${(comparisonResult.shared_concepts || []).join(', ')}\n\n`;
    md += `**Unique to ${comparisonResult.document_1.original_filename}:** ${(comparisonResult.unique_to_doc1 || []).join(', ')}\n\n`;
    md += `**Unique to ${comparisonResult.document_2.original_filename}:** ${(comparisonResult.unique_to_doc2 || []).join(', ')}\n\n`;

    md += `## 5. Contradictions & Disagreements\n`;
    if (!comparisonResult.contradictions || comparisonResult.contradictions.length === 0) {
      md += `No direct contradictions found between the documents.\n\n`;
    } else {
      comparisonResult.contradictions.forEach((c, idx) => {
        md += `### Conflict ${idx + 1}: ${c.topic}\n`;
        md += `- **${c.doc1_source}:** ${c.doc1_claim}\n`;
        md += `- **${c.doc2_source}:** ${c.doc2_claim}\n`;
        md += `*Analysis:* ${c.analysis}\n\n`;
      });
    }

    navigator.clipboard.writeText(md);
    setCopiedType('markdown');
    setTimeout(() => setCopiedType(null), 2000);
  };

  const handleCopyJson = () => {
    if (!comparisonResult) return;
    navigator.clipboard.writeText(JSON.stringify(comparisonResult, null, 2));
    setCopiedType('json');
    setTimeout(() => setCopiedType(null), 2000);
  };

  return (
    <div className="comparison-page">
      {/* Header */}
      <div className="comparison-header">
        <div className="comparison-title-box">
          <h1>
            <GitCompare size={24} style={{ color: 'var(--primary-600)' }} />
            Document Comparison & Cross-Document Analysis
          </h1>
          <p>
            Select any two documents to evaluate similarities, structural divergences, shared/unique concepts, and conflicting claims using Groq GPT-OSS 120B.
          </p>
        </div>
      </div>

      {/* Dual Document Selector Card */}
      <div className="comparison-selectors-card">
        <div className="comparison-dual-selector">
          {/* Document 1 Selector */}
          <div className="doc-selector-box">
            <label>
              <FileText size={14} /> Document 1 (Base Document)
            </label>
            <select value={docId1} onChange={(e) => setDocId1(e.target.value)}>
              <option value="">Select Document 1...</option>
              {availableDocs.map((doc) => (
                <option key={doc.id} value={doc.id} disabled={doc.id === docId2}>
                  {doc.original_filename || doc.filename}
                </option>
              ))}
            </select>
            {selectedDoc1 && (
              <div className="doc-meta-chips">
                <span className="doc-chip badge-primary">
                  {(selectedDoc1.file_type || '').toUpperCase()}
                </span>
                <span className="doc-chip">{selectedDoc1.size_formatted || 'N/A'}</span>
                <span className="doc-chip">
                  {selectedDoc1.chunks_count || selectedDoc1.chunk_count || 0} chunks
                </span>
                {selectedDoc1.pages && (
                  <span className="doc-chip">{selectedDoc1.pages} pages</span>
                )}
              </div>
            )}
          </div>

          {/* VS Badge */}
          <div className="comparison-vs-badge">
            <div className="vs-circle">VS</div>
          </div>

          {/* Document 2 Selector */}
          <div className="doc-selector-box">
            <label>
              <FileText size={14} /> Document 2 (Comparison Document)
            </label>
            <select value={docId2} onChange={(e) => setDocId2(e.target.value)}>
              <option value="">Select Document 2...</option>
              {availableDocs.map((doc) => (
                <option key={doc.id} value={doc.id} disabled={doc.id === docId1}>
                  {doc.original_filename || doc.filename}
                </option>
              ))}
            </select>
            {selectedDoc2 && (
              <div className="doc-meta-chips">
                <span className="doc-chip badge-primary">
                  {(selectedDoc2.file_type || '').toUpperCase()}
                </span>
                <span className="doc-chip">{selectedDoc2.size_formatted || 'N/A'}</span>
                <span className="doc-chip">
                  {selectedDoc2.chunks_count || selectedDoc2.chunk_count || 0} chunks
                </span>
                {selectedDoc2.pages && (
                  <span className="doc-chip">{selectedDoc2.pages} pages</span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Options Row (Focus Area + Action Button) */}
        <div className="comparison-options-row">
          <div className="focus-area-input-wrapper">
            <label htmlFor="focusAreaInput">
              Comparison Focus Area (Optional)
            </label>
            <input
              id="focusAreaInput"
              type="text"
              placeholder="e.g., Performance benchmarks, Security architecture, Pricing models, SLA"
              value={focusArea}
              onChange={(e) => setFocusArea(e.target.value)}
              disabled={isLoading}
            />
          </div>

          <button
            className="btn-compare-action"
            onClick={handleCompare}
            disabled={!canCompare}
          >
            <GitCompare size={18} />
            {isLoading ? 'Analyzing Documents...' : 'Compare Documents'}
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div
          style={{
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            color: '#991b1b',
            padding: '14px 18px',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            fontSize: '0.9rem',
          }}
        >
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="comparison-loading-card">
          <div className="loading-spinner-ring" />
          <h3>Cross-Document Analysis in Progress</h3>
          <p>
            Extracting representative sections, aligning thematic topics, and evaluating similarities and potential contradictions using Groq GPT-OSS 120B...
          </p>
        </div>
      )}

      {/* Comparison Results */}
      {comparisonResult && !isLoading && (
        <div className="comparison-results-container">
          {/* Metadata Banner */}
          <div className="results-meta-banner">
            <div className="results-docs-compared">
              <span className="doc-name-tag">
                {comparisonResult.document_1.original_filename}
              </span>
              <ArrowRightLeft size={14} style={{ color: 'var(--text-tertiary)' }} />
              <span className="doc-name-tag">
                {comparisonResult.document_2.original_filename}
              </span>
            </div>

            <div className="results-meta-stats">
              <span className="stat-pill">
                <Sparkles size={13} style={{ color: 'var(--primary-600)' }} />
                {comparisonResult.model}
              </span>
              <span className="stat-pill">
                <Clock size={13} />
                {(comparisonResult.processing_time_ms / 1000).toFixed(2)}s
              </span>
              <span className="stat-pill">
                <Layers size={13} />
                {comparisonResult.citations?.length || 0} chunks cited
              </span>

              <button
                className="btn-secondary-action"
                onClick={handleCopyMarkdown}
                title="Copy as formatted Markdown"
              >
                {copiedType === 'markdown' ? <Check size={14} color="#16a34a" /> : <Copy size={14} />}
                Markdown
              </button>
              <button
                className="btn-secondary-action"
                onClick={handleCopyJson}
                title="Copy as JSON"
              >
                {copiedType === 'json' ? <Check size={14} color="#16a34a" /> : <Copy size={14} />}
                JSON
              </button>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="comparison-tabs">
            {TABS.map((tab) => {
              const isContradictionTab = tab.id === 'contradictions';
              const contradictionCount = comparisonResult.contradictions?.length || 0;
              return (
                <button
                  key={tab.id}
                  className={`comparison-tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                  {isContradictionTab && contradictionCount > 0 && (
                    <span className="tab-badge badge-alert">{contradictionCount}</span>
                  )}
                  {tab.id === 'findings' && (
                    <span className="tab-badge">
                      {(comparisonResult.similarities?.length || 0) +
                        (comparisonResult.differences?.length || 0)}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Tab 1: Executive Summary */}
          {activeTab === 'summary' && (
            <div className="tab-content-panel">
              <div className="summary-narrative-box">
                {comparisonResult.comparison_summary}
              </div>
            </div>
          )}

          {/* Tab 2: Similarities & Differences */}
          {activeTab === 'findings' && (
            <div className="tab-content-panel">
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Identified Similarities ({comparisonResult.similarities?.length || 0})
              </h3>
              <div className="findings-grid">
                {(comparisonResult.similarities || []).map((item, idx) => (
                  <div key={idx} className="finding-card">
                    <div className="finding-card-header">
                      <span className="finding-topic">{item.topic}</span>
                      <span className="finding-source-tag">
                        {item.source_document || 'Both Documents'}
                        {item.source_pages ? ` • ${item.source_pages}` : ''}
                      </span>
                    </div>
                    <p className="finding-desc">{item.description}</p>
                    {item.excerpt && (
                      <div className="finding-excerpt">"{item.excerpt}"</div>
                    )}
                  </div>
                ))}
              </div>

              <h3
                style={{
                  fontSize: '1rem',
                  fontWeight: 700,
                  color: 'var(--text-primary)',
                  marginTop: '16px',
                }}
              >
                Identified Differences ({comparisonResult.differences?.length || 0})
              </h3>
              <div className="findings-grid">
                {(comparisonResult.differences || []).map((item, idx) => (
                  <div key={idx} className="finding-card">
                    <div className="finding-card-header">
                      <span className="finding-topic">{item.topic}</span>
                      <span className="finding-source-tag">
                        {item.source_document || 'Divergence'}
                        {item.source_pages ? ` • ${item.source_pages}` : ''}
                      </span>
                    </div>
                    <p className="finding-desc">{item.description}</p>
                    {item.excerpt && (
                      <div className="finding-excerpt">"{item.excerpt}"</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tab 3: Concepts & Themes */}
          {activeTab === 'concepts' && (
            <div className="tab-content-panel">
              <div className="concepts-section-grid">
                {/* Shared Concepts */}
                <div className="concept-category-card">
                  <div className="concept-category-title">
                    <CheckCircle2 size={16} color="var(--primary-600)" />
                    Shared Concepts ({comparisonResult.shared_concepts?.length || 0})
                  </div>
                  <div className="concept-pills-wrap">
                    {(comparisonResult.shared_concepts || []).map((c, i) => (
                      <span key={i} className="concept-pill shared">
                        {c}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Unique to Doc 1 */}
                <div className="concept-category-card">
                  <div className="concept-category-title">
                    <FileText size={16} color="#166534" />
                    Unique to {comparisonResult.document_1.original_filename} (
                    {comparisonResult.unique_to_doc1?.length || 0})
                  </div>
                  <div className="concept-pills-wrap">
                    {(comparisonResult.unique_to_doc1 || []).map((c, i) => (
                      <span key={i} className="concept-pill unique-doc1">
                        {c}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Unique to Doc 2 */}
                <div className="concept-category-card">
                  <div className="concept-category-title">
                    <FileText size={16} color="#1e40af" />
                    Unique to {comparisonResult.document_2.original_filename} (
                    {comparisonResult.unique_to_doc2?.length || 0})
                  </div>
                  <div className="concept-pills-wrap">
                    {(comparisonResult.unique_to_doc2 || []).map((c, i) => (
                      <span key={i} className="concept-pill unique-doc2">
                        {c}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab 4: Contradictions & Disagreements */}
          {activeTab === 'contradictions' && (
            <div className="tab-content-panel">
              {(!comparisonResult.contradictions ||
                comparisonResult.contradictions.length === 0) ? (
                <div className="no-contradictions-card">
                  <CheckCircle2 size={32} />
                  <div>
                    <h4>No Direct Contradictions Detected</h4>
                    <p>
                      Both documents exhibit coherent alignment across evaluated claims, data points, and conclusions without contradictory statements.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="contradictions-wrapper">
                  {comparisonResult.contradictions.map((c, idx) => (
                    <div key={idx} className="contradiction-card">
                      <div className="contradiction-header">
                        <AlertTriangle size={18} />
                        <span>Conflict {idx + 1}: {c.topic}</span>
                      </div>

                      <div className="contradiction-columns">
                        {/* Doc 1 Claim */}
                        <div className="claim-box">
                          <div className="claim-doc-title">
                            <span>{c.doc1_source}</span>
                            {c.doc1_pages && <span>Page {c.doc1_pages}</span>}
                          </div>
                          <p className="claim-text">{c.doc1_claim}</p>
                          {c.doc1_excerpt && (
                            <div className="finding-excerpt">"{c.doc1_excerpt}"</div>
                          )}
                        </div>

                        {/* Doc 2 Claim */}
                        <div className="claim-box">
                          <div className="claim-doc-title">
                            <span>{c.doc2_source}</span>
                            {c.doc2_pages && <span>Page {c.doc2_pages}</span>}
                          </div>
                          <p className="claim-text">{c.doc2_claim}</p>
                          {c.doc2_excerpt && (
                            <div className="finding-excerpt">"{c.doc2_excerpt}"</div>
                          )}
                        </div>
                      </div>

                      <div className="contradiction-analysis">
                        <strong>Analytical Synthesis:</strong> {c.analysis}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Tab 5: Agreements */}
          {activeTab === 'agreements' && (
            <div className="tab-content-panel">
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Reinforced Agreements & Consensus ({comparisonResult.agreements?.length || 0})
              </h3>
              <div className="findings-grid">
                {(comparisonResult.agreements || []).map((item, idx) => (
                  <div key={idx} className="finding-card">
                    <div className="finding-card-header">
                      <span className="finding-topic">{item.topic}</span>
                      <span className="finding-source-tag">
                        {item.source_document || 'Both Documents'}
                        {item.source_pages ? ` • ${item.source_pages}` : ''}
                      </span>
                    </div>
                    <p className="finding-desc">{item.description}</p>
                    {item.excerpt && (
                      <div className="finding-excerpt">"{item.excerpt}"</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tab 6: Citations */}
          {activeTab === 'citations' && (
            <div className="tab-content-panel">
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Grounded Document Chunks ({comparisonResult.citations?.length || 0})
              </h3>
              <div className="citations-grid">
                {(comparisonResult.citations || []).map((cit, idx) => (
                  <div key={idx} className="citation-card">
                    <div className="citation-header-row">
                      <span>{cit.document_name}</span>
                      <span className="citation-chunk-badge">
                        Chunk #{cit.chunk_index + 1}
                        {cit.source_pages ? ` • Page ${cit.source_pages}` : ''}
                        {cit.slide_number ? ` • Slide ${cit.slide_number}` : ''}
                      </span>
                    </div>
                    <div className="citation-excerpt">"{cit.excerpt}"</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DocumentComparisonPage;

