import React from 'react';
import { Info, Sparkles } from 'lucide-react';

const StageNotice = ({
  title = 'Stage 1 Architecture Preview',
  message = 'Full UI components and FastAPI backend foundation are active. In Stage 2, LangChain/ChromaDB and LLM embedding retrieval will be connected.',
  icon: Icon = Info,
}) => {
  return (
    <div className="stage-banner">
      <Icon size={18} className="stage-banner-icon" />
      <div>
        <div className="stage-banner-title">{title}</div>
        <div className="stage-banner-text">{message}</div>
      </div>
    </div>
  );
};

export default StageNotice;
