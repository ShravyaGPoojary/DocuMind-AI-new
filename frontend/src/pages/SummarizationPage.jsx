import React, { useState, useEffect, useRef } from 'react';
import {
  FileText,
  Copy,
  Download,
  Sparkles,
  Check,
  AlignLeft,
  ListFilter,
  Plus,
  AlertCircle,
  Cpu,
  Layers,
  FileCheck,
  RotateCcw,
} from 'lucide-react';
import Button from '../components/common/Button';
import Card, { CardHeader, CardBody } from '../components/common/Card';
import Badge from '../components/common/Badge';
import EmptyState from '../components/common/EmptyState';
import { useDocuments } from '../context/DocumentContext';
import {
  startDocumentSummaryJob,
  getSummaryJobStatus,
  resumeSummaryJob,
} from '../services/api';

const SUMMARY_TYPES = [
  {
    id: 'brief',
    title: 'Brief Summary',
    description: 'Concise high-level overview highlighting core findings and immediate takeaways.',
  },
  {
    id: 'detailed',
    title: 'Detailed Analytical Summary',
    description: 'Comprehensive section-by-section breakdown of main arguments, evidence, and conclusions.',
  },
  {
    id: 'key_points',
    title: 'Key Points & Insights',
    description: 'Bulleted takeaways highlighting critical facts, metrics, and actionable items.',
  },
  {
    id: 'executive',
    title: 'Executive Briefing',
    description: 'High-level strategic briefing tailored for decision-makers and leadership.',
  },
];

const SummarizationPage = () => {
  const {
    documents,
    activeDocument,
    activeDocumentId,
    setActiveDocumentId,
    setIsUploadModalOpen,
    showToast,
  } = useDocuments();

  const [selectedType, setSelectedType] = useState('brief');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedSummary, setGeneratedSummary] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [copied, setCopied] = useState(false);

  // Background Job & Progress State
  const [activeJobId, setActiveJobId] = useState(null);
  const [jobProgress, setJobProgress] = useState(null);
  const pollIntervalRef = useRef(null);

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const pollJob = (jobId) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    pollIntervalRef.current = setInterval(async () => {
      try {
        const statusData = await getSummaryJobStatus(jobId);
        setJobProgress({
          status: statusData.status,
          strategy: statusData.strategy,
          progressPercent: statusData.progress_percent,
          progressMessage: statusData.progress_message,
          totalChunks: statusData.total_chunks,
          processedChunks: statusData.processed_chunks,
          completedBatches: statusData.completed_batches,
          totalBatches: statusData.total_batches,
          currentBatch: statusData.current_batch,
          currentStage: statusData.current_stage || 'map',
        });

        if (statusData.status === 'completed') {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
          setIsGenerating(false);

          if (statusData.result) {
            const res = statusData.result;
            setGeneratedSummary({
              type: res.summary_type,
              typeLabel: res.summary_type_label,
              document: res.document_name,
              date: new Date().toLocaleDateString(),
              model: res.model,
              provider: res.provider,
              chunkCount: res.chunk_count,
              wordCount: res.word_count,
              strategyUsed: res.strategy_used,
              sections: res.sections || [],
              keyPoints: res.key_points || [],
              summary: res.summary || '',
            });
            showToast(`Generated ${res.summary_type_label} for "${statusData.document_name}"`, 'success');
          }
        } else if (statusData.status === 'failed') {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
          setIsGenerating(false);
          const errDetail = statusData.error_message || 'Summarization paused or failed.';
          setErrorMessage(errDetail);
          showToast(errDetail, 'error');
        }
      } catch (err) {
        console.error('Error polling summarization job:', err);
      }
    }, 1500);
  };

  const handleGenerate = async () => {
    if (!activeDocument) return;

    setIsGenerating(true);
    setErrorMessage(null);
    setGeneratedSummary(null);

    try {
      const initialJob = await startDocumentSummaryJob(activeDocument.id, {
        summaryType: selectedType,
      });

      setActiveJobId(initialJob.job_id);
      setJobProgress({
        status: initialJob.status,
        strategy: initialJob.strategy,
        progressPercent: initialJob.progress_percent || 5,
        progressMessage: initialJob.progress_message || 'Initializing summarization...',
        completedBatches: initialJob.completed_batches || 0,
        totalBatches: initialJob.total_batches || 1,
        currentBatch: initialJob.current_batch || 1,
      });

      pollJob(initialJob.job_id);
    } catch (err) {
      console.error('Failed to start summarization job:', err);
      setIsGenerating(false);
      let errorDetail =
        err.response?.data?.detail ||
        err.message ||
        'Failed to start document summarization.';

      if (err.response?.status === 401) {
        errorDetail = 'Groq API authentication failed. Please verify GROQ_API_KEY in backend/.env.';
      } else if (err.response?.status === 429) {
        errorDetail = 'Groq API rate limit reached. Please wait a moment before trying again.';
      } else if (err.response?.status === 504) {
        errorDetail = 'Summarization request timed out. Please try a briefer summary type.';
      } else if (err.response?.status === 503) {
        errorDetail = 'Network error reaching Groq API. Please verify your connection.';
      }

      setErrorMessage(errorDetail);
      showToast(errorDetail, 'error');
    }
  };

  const handleResume = async () => {
    if (!activeJobId) return;

    setIsGenerating(true);
    setErrorMessage(null);

    try {
      const resumedJob = await resumeSummaryJob(activeJobId);
      setJobProgress({
        status: resumedJob.status,
        strategy: resumedJob.strategy,
        progressPercent: resumedJob.progress_percent,
        progressMessage: resumedJob.progress_message,
        completedBatches: resumedJob.completed_batches,
        totalBatches: resumedJob.total_batches,
        currentBatch: resumedJob.current_batch,
      });
      showToast(`Resuming summarization from batch ${resumedJob.completed_batches + 1}...`, 'info');
      pollJob(activeJobId);
    } catch (err) {
      console.error('Failed to resume summarization job:', err);
      setIsGenerating(false);
      const errDetail = err.response?.data?.detail || err.message || 'Failed to resume job.';
      setErrorMessage(errDetail);
      showToast(errDetail, 'error');
    }
  };

  const handleCopy = () => {
    if (!generatedSummary) return;

    let text = generatedSummary.summary;
    if (!text && generatedSummary.sections) {
      text = `# ${generatedSummary.typeLabel}: ${generatedSummary.document}\n\n`;
      generatedSummary.sections.forEach((sec) => {
        text += `## ${sec.title}\n`;
        if (sec.content) text += `${sec.content}\n\n`;
        if (sec.bullets && sec.bullets.length > 0) {
          sec.bullets.forEach((b) => (text += `- ${b}\n`));
          text += '\n';
        }
      });
    }

    navigator.clipboard.writeText(text);
    setCopied(true);
    showToast('Summary copied to clipboard!', 'info');
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!generatedSummary) return;

    let text = generatedSummary.summary;
    if (!text && generatedSummary.sections) {
      text = `# ${generatedSummary.typeLabel}: ${generatedSummary.document}\n\n`;
      generatedSummary.sections.forEach((sec) => {
        text += `## ${sec.title}\n`;
        if (sec.content) text += `${sec.content}\n\n`;
        if (sec.bullets && sec.bullets.length > 0) {
          sec.bullets.forEach((b) => (text += `- ${b}\n`));
          text += '\n';
        }
      });
    }

    const headerNote = `<!-- Generated by DocuMind AI (${generatedSummary.model}) on ${generatedSummary.date} -->\n\n`;
    const blob = new Blob([headerNote + text], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `DocuMind_Summary_${activeDocument?.name || 'document'}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    showToast('Markdown file downloaded!', 'success');
  };

  if (documents.length === 0) {
    return (
      <div className="summarization-page">
        <div className="page-header-row">
          <div className="page-header-title-group">
            <h1>Document Summarization</h1>
            <p>Generate grounded executive briefings or detailed analytical breakdowns.</p>
          </div>
        </div>
        <EmptyState
          icon={AlignLeft}
          title="No documents to summarize"
          description="Upload a document to the library first before generating a summary."
          actionText="Upload Document"
          actionIcon={Plus}
          onAction={() => setIsUploadModalOpen(true)}
        />
      </div>
    );
  }

  return (
    <div className="summarization-page">
      <div className="page-header-row">
        <div className="page-header-title-group">
          <h1>Document Summarization</h1>
          <p>
            Generate intelligent, grounded summaries using Groq GPT-OSS 120B across any document format.
          </p>
        </div>
      </div>

      {/* Global Error & Resume Banner */}
      {errorMessage && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '12px 16px',
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: 'var(--radius-md)',
            color: '#991b1b',
            fontSize: '0.875rem',
            gap: '12px',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1 }}>
            <AlertCircle size={18} style={{ flexShrink: 0 }} />
            <span>{errorMessage}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {activeJobId && (
              <Button
                variant="secondary"
                size="sm"
                icon={RotateCcw}
                onClick={handleResume}
              >
                {jobProgress?.completedBatches > 0
                  ? `Resume from Batch ${jobProgress.completedBatches + 1}`
                  : 'Resume Job'}
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={() => setErrorMessage(null)}>
              Dismiss
            </Button>
          </div>
        </div>
      )}

      <div className="summarization-grid">
        {/* Left Control Panel */}
        <Card className="summary-control-panel">
          <CardHeader title="Configuration" icon={ListFilter} />
          <CardBody style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Target Document Selector */}
            <div className="form-group">
              <label className="form-label">Target Document</label>
              <select
                className="form-select"
                value={activeDocumentId || activeDocument?.id || ''}
                onChange={(e) => {
                  setActiveDocumentId(e.target.value);
                  setGeneratedSummary(null);
                  setErrorMessage(null);
                  setActiveJobId(null);
                  setJobProgress(null);
                }}
                disabled={isGenerating}
              >
                {documents.map((doc) => (
                  <option key={doc.id} value={doc.id}>
                    {doc.name} ({doc.size_formatted})
                  </option>
                ))}
              </select>
            </div>

            {/* Summary Format Selection */}
            <div className="form-group">
              <label className="form-label">Summary Type</label>
              <div className="summary-type-cards">
                {SUMMARY_TYPES.map((type) => (
                  <div
                    key={type.id}
                    className={`type-option-card ${selectedType === type.id ? 'selected' : ''}`}
                    onClick={() => !isGenerating && setSelectedType(type.id)}
                    style={{ cursor: isGenerating ? 'not-allowed' : 'pointer' }}
                  >
                    <div className="type-radio-circle">
                      {selectedType === type.id && <div className="type-radio-dot" />}
                    </div>
                    <div className="type-option-content">
                      <span className="type-option-title">{type.title}</span>
                      <span className="type-option-desc">{type.description}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Provider notice */}
            <div
              style={{
                fontSize: '0.75rem',
                color: 'var(--text-tertiary)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '8px 12px',
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-md)',
              }}
            >
              <Cpu size={14} style={{ color: 'var(--primary-600)' }} />
              <span>Powered by <strong>Groq GPT-OSS 120B</strong></span>
            </div>

            {/* Generate & Resume Buttons */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <Button
                variant="primary"
                size="lg"
                icon={Sparkles}
                loading={isGenerating}
                disabled={!activeDocument || isGenerating}
                onClick={handleGenerate}
              >
                {isGenerating ? 'Synthesizing Summary...' : 'Generate Summary'}
              </Button>

              {activeJobId && !isGenerating && jobProgress?.status === 'failed' && (
                <Button
                  variant="secondary"
                  size="md"
                  icon={RotateCcw}
                  onClick={handleResume}
                >
                  {jobProgress?.completedBatches > 0
                    ? `Resume from Batch ${jobProgress.completedBatches + 1}`
                    : 'Resume Summarization'}
                </Button>
              )}
            </div>
          </CardBody>
        </Card>

        {/* Right Output Area */}
        <div className="summary-output-panel">
          {isGenerating ? (
            <div className="summary-progress-container">
              <div style={{ position: 'relative' }}>
                <Sparkles
                  size={36}
                  style={{ color: 'var(--primary-600)', animation: 'spin 3s linear infinite' }}
                />
              </div>

              <h3 style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                {jobProgress?.strategy === 'map_reduce'
                  ? 'Adaptive Token-Aware Map-Reduce Summarization'
                  : 'Synthesizing Intelligent Summary...'}
              </h3>

              {jobProgress?.strategy === 'map_reduce' && (
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
                  <div className="summary-batch-badge">
                    <Layers size={13} />
                    <span>
                      Adaptive Batch {jobProgress.currentBatch || jobProgress.completedBatches + 1} of {jobProgress.totalBatches || 1}
                      {jobProgress.completedBatches > 0 ? ` (${jobProgress.completedBatches} completed)` : ''}
                    </span>
                  </div>

                  {jobProgress.totalChunks > 0 && (
                    <div className="summary-batch-badge" style={{ backgroundColor: '#f0fdf4', borderColor: '#bbf7d0', color: '#166534' }}>
                      <FileCheck size={13} />
                      <span>
                        {jobProgress.processedChunks || 0} / {jobProgress.totalChunks} Chunks Covered
                      </span>
                    </div>
                  )}

                  <div className="summary-batch-badge" style={{ backgroundColor: '#faf5ff', borderColor: '#e9d5ff', color: '#6b21a8' }}>
                    <Cpu size={13} />
                    <span>
                      Stage: {jobProgress.currentStage === 'reduce' ? 'Hierarchical Reduction' : 'Map Batching'}
                    </span>
                  </div>
                </div>
              )}

              {/* Real-time Progress Bar */}
              <div style={{ width: '100%', maxWidth: '440px', marginTop: '8px' }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '0.75rem',
                    color: 'var(--text-secondary)',
                    marginBottom: '4px',
                  }}
                >
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '360px' }}>
                    {jobProgress?.progressMessage || 'Analyzing document chunks...'}
                  </span>
                  <span><strong>{jobProgress?.progressPercent || 0}%</strong></span>
                </div>
                <div className="summary-progress-bar-track">
                  <div
                    className="summary-progress-bar-fill"
                    style={{ width: `${Math.max(jobProgress?.progressPercent || 5, 5)}%` }}
                  />
                </div>
              </div>

              <p
                style={{
                  fontSize: '0.8125rem',
                  color: 'var(--text-tertiary)',
                  maxWidth: '460px',
                  lineHeight: 1.5,
                  marginTop: '8px',
                }}
              >
                Dynamically budgeted within ~6,000 tokens per request to comply with Groq Free Tier limits. Intermediate digests are cached in SQLite for instant resumability.
              </p>
            </div>
          ) : generatedSummary ? (
            <div className="summary-content-paper">
              <div className="summary-paper-header">
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '16px' }}>
                  <div>
                    <h2 className="summary-target-title">{generatedSummary.document}</h2>
                    <div className="summary-meta-row" style={{ flexWrap: 'wrap', gap: '8px', marginTop: '6px' }}>
                      <Badge variant="primary">{generatedSummary.typeLabel}</Badge>
                      <Badge variant="outline">{generatedSummary.model}</Badge>
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          fontSize: '0.75rem',
                          color: 'var(--text-secondary)',
                        }}
                      >
                        <Layers size={12} />
                        {generatedSummary.strategyUsed === 'map_reduce'
                          ? 'Hierarchical Map-Reduce'
                          : 'Direct Single-Pass'}
                      </span>
                      <span>• {generatedSummary.chunkCount} chunks</span>
                      <span>• {generatedSummary.wordCount} words</span>
                    </div>
                  </div>

                  <div className="output-header-actions">
                    <Button
                      variant="secondary"
                      size="sm"
                      icon={copied ? Check : Copy}
                      onClick={handleCopy}
                    >
                      {copied ? 'Copied' : 'Copy'}
                    </Button>
                    <Button
                      variant="primary"
                      size="sm"
                      icon={Download}
                      onClick={handleDownload}
                    >
                      Download .MD
                    </Button>
                  </div>
                </div>
              </div>

              {/* Prose Body */}
              <div className="summary-prose-body">
                {generatedSummary.sections.map((section, sIdx) => (
                  <div key={sIdx} style={{ marginBottom: '16px' }}>
                    <h3 className="summary-section-title">{section.title}</h3>
                    {section.content && (
                      <p style={{ marginTop: '6px', whiteSpace: 'pre-wrap' }}>{section.content}</p>
                    )}
                    {section.bullets && section.bullets.length > 0 && (
                      <ul className="summary-bullets" style={{ marginTop: '8px' }}>
                        {section.bullets.map((b, bIdx) => (
                          <li key={bIdx}>{b}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <EmptyState
              icon={AlignLeft}
              title="No summary generated yet"
              description={`Select a summary depth on the left and click "Generate Summary" to process "${activeDocument?.name}".`}
              actionText="Generate Summary"
              actionIcon={Sparkles}
              onAction={handleGenerate}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default SummarizationPage;
