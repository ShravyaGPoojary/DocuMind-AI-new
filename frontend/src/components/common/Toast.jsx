import React from 'react';
import { CheckCircle2, AlertCircle, Info } from 'lucide-react';
import { useDocuments } from '../../context/DocumentContext';

const Toast = () => {
  const { toast } = useDocuments();

  if (!toast) return null;

  const Icon =
    toast.type === 'success' ? CheckCircle2 : toast.type === 'error' ? AlertCircle : Info;

  return (
    <div className="toast-container">
      <div className={`toast toast-${toast.type}`}>
        <Icon size={18} />
        <span>{toast.message}</span>
      </div>
    </div>
  );
};

export default Toast;
