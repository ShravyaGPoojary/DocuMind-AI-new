import React from 'react';

const Badge = ({ children, variant = 'draft', className = '', ...props }) => {
  return (
    <span className={`badge badge-${variant} ${className}`} {...props}>
      {children}
    </span>
  );
};

export const FileTypeBadge = ({ type = 'PDF' }) => {
  const normalized = (type || 'PDF').toLowerCase();
  return (
    <span className={`badge-filetype ${normalized}`}>
      .{normalized}
    </span>
  );
};

export default Badge;
