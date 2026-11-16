import React from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { Search, Menu, ChevronRight, User } from 'lucide-react';
import { useDocuments } from '../../context/DocumentContext';

const PAGE_METADATA = {
  '/': { title: 'Dashboard', breadcrumb: 'Overview' },
  '/documents': { title: 'Document Library', breadcrumb: 'Library' },
  '/chat': { title: 'AI Chat', breadcrumb: 'Chat' },
  '/summarize': { title: 'Summarization', breadcrumb: 'Summaries' },
  '/multi-doc': { title: 'Multi-Document Q&A', breadcrumb: 'Synthesis' },
  '/settings': { title: 'Settings', breadcrumb: 'Configuration' },
};

const Header = ({ onOpenMobileMenu }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { searchQuery, setSearchQuery } = useDocuments();

  const currentPage = PAGE_METADATA[location.pathname] || {
    title: 'DocuMind AI',
    breadcrumb: 'DocuMind',
  };

  const handleSearchKeyDown = (e) => {
    if (e.key === 'Enter' && location.pathname !== '/documents') {
      navigate('/documents');
    }
  };

  return (
    <header className="top-header">
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        {/* Mobile menu toggle */}
        <button
          className="header-action-btn mobile-menu-btn"
          onClick={onOpenMobileMenu}
          aria-label="Open navigation menu"
        >
          <Menu size={18} />
        </button>

        <div className="header-left">
          <div className="breadcrumbs">
            <Link to="/">DocuMind</Link>
            <ChevronRight size={12} className="breadcrumb-separator" />
            <span className="breadcrumb-current">{currentPage.breadcrumb}</span>
          </div>
          <h1 className="page-title">{currentPage.title}</h1>
        </div>
      </div>

      <div className="header-right">
        {/* Document Search input */}
        <div className="header-search">
          <Search size={15} className="header-search-icon" />
          <input
            type="text"
            className="header-search-input"
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleSearchKeyDown}
          />
        </div>

        {/* Minimal Profile Icon */}
        <div className="user-icon-btn" title="User Profile">
          <User size={18} />
        </div>
      </div>
    </header>
  );
};

export default Header;
