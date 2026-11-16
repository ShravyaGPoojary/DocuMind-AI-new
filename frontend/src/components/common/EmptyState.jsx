import React from 'react';
import { FolderOpen } from 'lucide-react';
import Button from './Button';

const EmptyState = ({
  icon: Icon = FolderOpen,
  title = 'No documents found',
  description = 'You have not uploaded any documents yet. Upload a PDF, DOCX, TXT, MD, or CSV file to begin.',
  actionText,
  actionIcon,
  onAction,
}) => {
  return (
    <div className="empty-state">
      <div className="empty-state-icon-box">
        <Icon size={26} />
      </div>
      <h3 className="empty-state-title">{title}</h3>
      <p className="empty-state-desc">{description}</p>
      {actionText && onAction && (
        <Button variant="primary" icon={actionIcon} onClick={onAction}>
          {actionText}
        </Button>
      )}
    </div>
  );
};

export default EmptyState;
