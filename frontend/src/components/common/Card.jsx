import React from 'react';

const Card = ({ children, className = '', hover = false, ...props }) => {
  return (
    <div className={`card ${hover ? 'card-hover' : ''} ${className}`} {...props}>
      {children}
    </div>
  );
};

export const CardHeader = ({ title, action, icon: Icon, className = '' }) => {
  return (
    <div className={`card-header ${className}`}>
      <div className="card-header-title">
        {Icon && <Icon size={18} className="text-muted" />}
        <span>{title}</span>
      </div>
      {action && <div className="card-header-action">{action}</div>}
    </div>
  );
};

export const CardBody = ({ children, className = '' }) => {
  return <div className={`card-body ${className}`}>{children}</div>;
};

export const CardFooter = ({ children, className = '' }) => {
  return <div className={`card-footer ${className}`}>{children}</div>;
};

export default Card;
