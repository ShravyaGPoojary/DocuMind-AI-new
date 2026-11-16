import React, { useState, useEffect } from 'react';
import {
  Brain,
  Sparkles,
  FileText,
  Calendar,
  Layers,
  HelpCircle,
  Compass,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Search,
  BookOpen,
  TrendingUp,
  Tag,
  Clock,
} from 'lucide-react';
import { useDocuments } from '../context/DocumentContext';
import {
  extractInformation,
  getDocumentInsights,
  generateQuestions,
  contextualAnalysis,
} from '../services/api';
import '../styles/intelligence.css';

const TABS = [
  { id: 'extraction', label: 'Structured Extraction', icon: Tag },
  { id: 'insights', label: 'Document Insights', icon: Compass },
  { id: 'questions', label: 'Knowledge-Check Q&A', icon: HelpCircle },
  { id: 'contextual', label: 'Contextual Reasoning', icon: Brain },
];

const DocumentIntelligencePage = () => {
  const { documents, fetchDocuments } = useDocuments();

  const [selectedDocId, setSelectedDocId] = useState('');
  const [activeTab, setActiveTab] = useState('extraction');

  // Intelligence State
  const [extractionData, setExtractionData] = useState(null);
  const [insightsData, setInsightsData] = useState(null);
  const [questionsData, setQuestionsData] = useState(null);
  const [contextualData, setContextualData] = useState(null);

  // Contextual Inquiry Input
  const [inquiryText, setInquiryText] = useState('');
  const [revealedAnswers, setRevealedAnswers] = useState({});

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!documents || documents.length === 0) {
      fetchDocuments();
    }
  }, [documents, fetchDocuments]);

  // Pre-select first ready document
  useEffect(() => {
    const readyDocs = (documents || []).filter(
      (d) => d.status === 'ready' || d.processing_status === 'ready' || (d.chunk_count && d.chunk_count > 0)
    );
    if (readyDocs.length > 0 && !selectedDocId) {
      setSelectedDocId(readyDocs[0].id);
    }
  }, [documents, selectedDocId]);

  const availableDocs = (documents || []).filter(
    (d) => d.status === 'ready' || d.processing_status === 'ready' || (d.chunk_count && d.chunk_count > 0)
  );

  const selectedDoc = availableDocs.find((d) => d.id === selectedDocId);

  // Trigger analysis based on active tab
  const handleAnalyze = async () => {
    if (!selectedDocId) return;
    setIsLoading(true);
    setError(null);

    try {
      if (activeTab === 'extraction') {
        const data = await extractInformation(selectedDocId);
        setExtractionData(data);
      } else if (activeTab === 'insights') {
        const data = await getDocumentInsights(selectedDocId);
        setInsightsData(data);
      } else if (activeTab === 'questions') {
        const data = await generateQuestions(selectedDocId, 5, 'mixed');
        setQuestionsData(data);
      } else if (activeTab === 'contextual') {
        if (!inquiryText.trim()) {
          setError('Please enter an inquiry, question, or hypothesis to perform contextual analysis.');
          setIsLoading(false);
          return;
        }
        const data = await contextualAnalysis(selectedDocId, inquiryText.trim());
        setContextualData(data);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Analysis failed. Please check Groq API key configuration.';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleRevealAnswer = (qId) => {
    setRevealedAnswers((prev) => ({ ...prev, [qId]: !prev[qId] }));
  };

  return (
    <div className="intelligence-page">
      {/* Page Header */}
      <div className="intelligence-header">
        <div className="intelligence-title-box">
          <h1>
            <Brain size={24} style={{ color: 'var(--primary-600)' }} />
            Advanced Document Intelligence
          </h1>
          <p>
            Extract structured facts, derive strategic insights, generate knowledge-check Q&A, and perform multi-hop contextual reasoning powered by Groq GPT-OSS 120B.
          </p>
        </div>
      </div>

      {/* Control Card: Document Selector & Mode Tabs */}
      <div className="intelligence-control-card">
        <div className="intelligence-doc-row">
          <div className="intelligence-doc-select">
            <label htmlFor="intelDocSelect">Select Document to Analyze</label>
            <select
              id="intelDocSelect"
              value={selectedDocId}
              onChange={(e) => {
                setSelectedDocId(e.target.value);
                setExtractionData(null);
                setInsightsData(null);
                setQuestionsData(null);
                setContextualData(null);
              }}
              disabled={isLoading}
            >
              <option value="">Choose an uploaded document...</option>
              {availableDocs.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.original_filename || doc.filename} ({(doc.file_type || '').toUpperCase()})
                </option>
              ))}
            </select>
          </div>

          <button
            className="btn-intelligence-action"
            onClick={handleAnalyze}
            disabled={!selectedDocId || isLoading}
          >
            <Sparkles size={18} />
            {isLoading ? 'Analyzing...' : `Run ${TABS.find((t) => t.id === activeTab)?.label}`}
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="intelligence-tabs">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={`intelligence-tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab(tab.id);
                  setError(null);
                }}
              >
                <Icon size={16} />
                {tab.label}
              </button>
            );
          })}
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

      {/* Loading Card */}
      {isLoading && (
        <div
          style={{
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-lg)',
            padding: '48px',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '16px',
          }}
        >
          <div
            style={{
              width: '44px',
              height: '44px',
              border: '3px solid var(--border-color)',
              borderTopColor: 'var(--primary-600)',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }}
          />
          <h3 style={{ fontSize: '1.1rem', color: 'var(--text-primary)' }}>Analyzing Document Intelligence</h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', maxWidth: '460px' }}>
            Processing document sections, extracting structured facts, and synthesizing results via Groq GPT-OSS 120B...
          </p>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* TAB 1: Structured Information Extraction                          */}
      {/* ------------------------------------------------------------------ */}
      {activeTab === 'extraction' && !isLoading && (
        <div className="intelligence-panel">
          {!extractionData ? (
            <div style={{ textAlign: 'center', padding: '32px', color: 'var(--text-secondary)' }}>
              Click <strong>Run Structured Extraction</strong> above to identify entities, dates, definitions, and key metrics.
            </div>
          ) : (
            <>
              {/* Entities */}
              <div className="panel-section-title">
                <Tag size={18} color="var(--primary-600)" />
                Extracted Entities ({extractionData.entities?.length || 0})
              </div>
              <div className="data-subgrid">
                {(extractionData.entities || []).map((ent, i) => (
                  <div key={i} className="data-card">
                    <div className="data-card-header">
                      <span className="data-card-title">{ent.name}</span>
                      <span className={`data-badge cat-${ent.category?.toLowerCase() || 'tech'}`}>
                        {ent.category}
                      </span>
                    </div>
                    <p className="data-desc">{ent.description}</p>
                    {ent.source_page && <span className="data-page-tag">📍 {ent.source_page}</span>}
                  </div>
                ))}
              </div>

              {/* Dates & Events */}
              <div className="panel-section-title">
                <Calendar size={18} color="#059669" />
                Key Dates & Timelines ({extractionData.dates_and_events?.length || 0})
              </div>
              <div className="data-subgrid">
                {(extractionData.dates_and_events || []).map((d, i) => (
                  <div key={i} className="data-card">
                    <div className="data-card-header">
                      <span className="data-card-title">{d.date_str}</span>
                    </div>
                    <p className="data-desc">{d.event_description}</p>
                    {d.source_page && <span className="data-page-tag">📍 {d.source_page}</span>}
                  </div>
                ))}
              </div>

              {/* Technical Definitions */}
              <div className="panel-section-title">
                <BookOpen size={18} color="#2563eb" />
                Technical Definitions & Terms ({extractionData.definitions?.length || 0})
              </div>
              <div className="data-subgrid">
                {(extractionData.definitions || []).map((def, i) => (
                  <div key={i} className="data-card">
                    <div className="data-card-header">
                      <span className="data-card-title">{def.term}</span>
                    </div>
                    <p className="data-desc">{def.definition}</p>
                    {def.context && (
                      <div style={{ fontSize: '0.8rem', fontStyle: 'italic', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                        "{def.context}"
                      </div>
                    )}
                    {def.source_page && <span className="data-page-tag">📍 {def.source_page}</span>}
                  </div>
                ))}
              </div>

              {/* Quantitative Metrics */}
              <div className="panel-section-title">
                <TrendingUp size={18} color="#d97706" />
                Critical Facts & Metrics ({extractionData.key_metrics?.length || 0})
              </div>
              <div className="data-subgrid">
                {(extractionData.key_metrics || []).map((m, i) => (
                  <div key={i} className="data-card">
                    <div className="data-card-header">
                      <span className="data-card-title">{m.metric_name}</span>
                      <span className="data-badge cat-metric">{m.value}</span>
                    </div>
                    <p className="data-desc">{m.context}</p>
                    {m.source_page && <span className="data-page-tag">📍 {m.source_page}</span>}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* TAB 2: Document Insights                                          */}
      {/* ------------------------------------------------------------------ */}
      {activeTab === 'insights' && !isLoading && (
        <div className="intelligence-panel">
          {!insightsData ? (
            <div style={{ textAlign: 'center', padding: '32px', color: 'var(--text-secondary)' }}>
              Click <strong>Run Document Insights</strong> above to generate executive synthesis, structural roadmap, and major findings.
            </div>
          ) : (
            <>
              {/* Executive Summary */}
              <div className="panel-section-title">
                <Compass size={18} color="var(--primary-600)" />
                Executive Analytical Synthesis
              </div>
              <div
                style={{
                  lineHeight: '1.7',
                  fontSize: '0.95rem',
                  color: 'var(--text-primary)',
                  backgroundColor: 'var(--bg-secondary)',
                  padding: '16px',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-color)',
                  whiteSpace: 'pre-wrap',
                }}
              >
                {insightsData.executive_summary}
              </div>

              {/* Key Topics */}
              <div className="panel-section-title" style={{ marginTop: '12px' }}>
                <Layers size={18} color="#6366f1" />
                Strategic Topics & Themes
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {(insightsData.key_topics || []).map((t, i) => (
                  <span
                    key={i}
                    style={{
                      padding: '4px 12px',
                      borderRadius: 'var(--radius-full)',
                      backgroundColor: 'var(--primary-50)',
                      color: 'var(--primary-700)',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      border: '1px solid var(--primary-200)',
                    }}
                  >
                    {t}
                  </span>
                ))}
              </div>

              {/* Major Findings & Risks */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '12px' }}>
                <div>
                  <div className="panel-section-title">
                    <CheckCircle2 size={18} color="#16a34a" />
                    Major Findings
                  </div>
                  <div className="findings-list">
                    {(insightsData.major_findings || []).map((f, i) => (
                      <div key={i} className="finding-bullet">
                        <span>•</span>
                        <span>{f}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="panel-section-title">
                    <AlertTriangle size={18} color="#dc2626" />
                    Risks & Challenges
                  </div>
                  <div className="findings-list">
                    {(insightsData.risks_and_challenges || []).map((r, i) => (
                      <div key={i} className="finding-bullet">
                        <span>•</span>
                        <span>{r}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Structural Roadmap */}
              {insightsData.structural_roadmap?.length > 0 && (
                <>
                  <div className="panel-section-title" style={{ marginTop: '16px' }}>
                    <FileText size={18} color="#4f46e5" />
                    Document Structural Roadmap
                  </div>
                  <div className="roadmap-timeline">
                    {insightsData.structural_roadmap.map((sec, i) => (
                      <div key={i} className="roadmap-item">
                        <h4>{sec.section_name}</h4>
                        <p>{sec.overview}</p>
                        {sec.key_takeaway && (
                          <div style={{ fontSize: '0.8rem', color: 'var(--primary-700)', fontWeight: 600, marginTop: '2px' }}>
                            Key Takeaway: {sec.key_takeaway}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* TAB 3: Knowledge-Check Question Generation                        */}
      {/* ------------------------------------------------------------------ */}
      {activeTab === 'questions' && !isLoading && (
        <div className="intelligence-panel">
          {!questionsData ? (
            <div style={{ textAlign: 'center', padding: '32px', color: 'var(--text-secondary)' }}>
              Click <strong>Run Knowledge-Check Q&A</strong> above to generate revision questions grounded in this document.
            </div>
          ) : (
            <div className="questions-list">
              <div className="panel-section-title">
                <HelpCircle size={18} color="var(--primary-600)" />
                Generated Revision Questions ({questionsData.questions?.length || 0})
              </div>
              {(questionsData.questions || []).map((q, idx) => {
                const isRevealed = !!revealedAnswers[q.id];
                return (
                  <div key={q.id} className="question-card">
                    <div className="question-header">
                      <span className="question-title">
                        {idx + 1}. {q.question}
                      </span>
                      <div className="question-meta-chips">
                        <span className={`diff-chip ${q.difficulty?.toLowerCase() || 'medium'}`}>
                          {q.difficulty}
                        </span>
                        <span className="data-badge">{q.question_type}</span>
                        {q.source_pages && <span className="data-badge">📍 {q.source_pages}</span>}
                      </div>
                    </div>

                    <button
                      className="btn-reveal-answer"
                      onClick={() => toggleRevealAnswer(q.id)}
                    >
                      {isRevealed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      {isRevealed ? 'Hide Answer' : 'Reveal Grounded Answer'}
                    </button>

                    {isRevealed && (
                      <div className="answer-box">
                        <div>{q.answer}</div>
                        {q.evidence_excerpt && (
                          <div className="evidence-quote">
                            <strong>Source Evidence:</strong> "{q.evidence_excerpt}"
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* TAB 4: Contextual Reasoning                                       */}
      {/* ------------------------------------------------------------------ */}
      {activeTab === 'contextual' && !isLoading && (
        <div className="intelligence-panel">
          <div className="contextual-input-box">
            <label htmlFor="inquiryText" style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Hypothesis, Query, or Proposition for Cross-Section Reasoning
            </label>
            <textarea
              id="inquiryText"
              placeholder="e.g. Does the proposed database architecture meet the SLA throughput requirements under high concurrency?"
              value={inquiryText}
              onChange={(e) => setInquiryText(e.target.value)}
              disabled={isLoading}
            />
          </div>

          {contextualData && (
            <>
              {/* Thesis Verdict */}
              <div className="panel-section-title" style={{ marginTop: '12px' }}>
                <Brain size={18} color="var(--primary-600)" />
                Thesis Assessment & Reasoning Verdict
              </div>
              <div
                style={{
                  backgroundColor: '#f8fafc',
                  border: '1px solid #cbd5e1',
                  borderLeft: '4px solid var(--primary-600)',
                  padding: '16px',
                  borderRadius: '0 var(--radius-md) var(--radius-md) 0',
                  fontSize: '0.95rem',
                  color: '#1e293b',
                  lineHeight: '1.6',
                }}
              >
                {contextualData.thesis_assessment}
              </div>

              {/* Supporting vs Counter Evidence */}
              <div className="contextual-results-grid" style={{ marginTop: '16px' }}>
                <div className="evidence-column supporting">
                  <h4 style={{ fontSize: '0.925rem', fontWeight: 700, color: '#15803d', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <CheckCircle2 size={16} /> Supporting Evidence ({contextualData.supporting_evidence?.length || 0})
                  </h4>
                  {(contextualData.supporting_evidence || []).map((ev, i) => (
                    <div key={i} className="evidence-item">
                      <strong>{ev.claim}</strong>
                      {ev.excerpt && <span style={{ fontStyle: 'italic', color: 'var(--text-tertiary)' }}>"{ev.excerpt}"</span>}
                      <span style={{ fontSize: '0.78rem', color: '#15803d' }}>Significance: {ev.significance}</span>
                      {ev.source_pages && <span className="data-page-tag">📍 {ev.source_pages}</span>}
                    </div>
                  ))}
                </div>

                <div className="evidence-column counter">
                  <h4 style={{ fontSize: '0.925rem', fontWeight: 700, color: '#c2410c', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <AlertTriangle size={16} /> Counter-Evidence & Limitations ({contextualData.counter_evidence_or_limitations?.length || 0})
                  </h4>
                  {(contextualData.counter_evidence_or_limitations || []).map((ev, i) => (
                    <div key={i} className="evidence-item">
                      <strong>{ev.claim}</strong>
                      {ev.excerpt && <span style={{ fontStyle: 'italic', color: 'var(--text-tertiary)' }}>"{ev.excerpt}"</span>}
                      <span style={{ fontSize: '0.78rem', color: '#c2410c' }}>Significance: {ev.significance}</span>
                      {ev.source_pages && <span className="data-page-tag">📍 {ev.source_pages}</span>}
                    </div>
                  ))}
                </div>
              </div>

              {/* Synthesis */}
              <div className="panel-section-title" style={{ marginTop: '16px' }}>
                <Compass size={18} color="var(--primary-600)" />
                Synthesis & Strategic Implications
              </div>
              <div
                style={{
                  lineHeight: '1.65',
                  fontSize: '0.9rem',
                  color: 'var(--text-secondary)',
                  backgroundColor: 'var(--bg-secondary)',
                  padding: '14px 16px',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-color)',
                }}
              >
                {contextualData.synthesis_and_implications}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default DocumentIntelligencePage;

