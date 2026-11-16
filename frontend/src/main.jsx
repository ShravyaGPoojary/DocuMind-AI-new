import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './styles/index.css';
import './styles/layout.css';
import './styles/components.css';
import './styles/dashboard.css';
import './styles/documents.css';
import './styles/chat.css';
import './styles/summarization.css';
import './styles/multidoc.css';
import './styles/settings.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
