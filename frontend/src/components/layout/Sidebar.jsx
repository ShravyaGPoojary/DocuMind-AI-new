import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Files,
  MessageSquare,
  FileText,
  Boxes,
  GitCompare,
  Brain,
  Settings,
  Github,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  LogOut,
  User as UserIcon,
} from 'lucide-react';
import { useDocuments } from '../../context/DocumentContext';
import { useAuth } from '../../context/AuthContext';

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/documents', label: 'Documents', icon: Files },
  { to: '/chat', label: 'AI Chat', icon: MessageSquare },
  { to: '/intelligence', label: 'Intelligence', icon: Brain },
  { to: '/summarize', label: 'Summarization', icon: FileText },
  { to: '/multi-doc', label: 'Multi-Doc Q&A', icon: Boxes },
  { to: '/compare', label: 'Comparison', icon: GitCompare },
  { to: '/settings', label: 'Settings', icon: Settings },
];


const Sidebar = ({ isCollapsed, onToggleCollapse, isMobileOpen, onCloseMobile }) => {
  const { backendStatus, backendInfo } = useDocuments();
  const { user, logout } = useAuth();
  const githubRepoUrl = import.meta.env.VITE_GITHUB_REPO_URL;

  const initials = user?.full_name
    ? user.full_name
        .split(' ')
        .map((n) => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2)
    : 'U';


  return (
    <aside
      className={`sidebar ${isCollapsed ? 'collapsed' : ''} ${isMobileOpen ? 'mobile-open' : ''}`}
    >
      {/* Sidebar Header / Brand */}
      <div className="sidebar-header">
        <NavLink to="/" className="brand-wrapper" onClick={onCloseMobile}>
          <div className="brand-icon-box">
            <Sparkles size={20} />
          </div>
          {!isCollapsed && (
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span className="brand-title">DocuMind</span>
              <span className="brand-tag">AI</span>
            </div>
          )}
        </NavLink>

        <button
          className="sidebar-collapse-btn"
          onClick={onToggleCollapse}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      {/* Main Navigation */}
      <nav className="sidebar-nav">
        {!isCollapsed && <div className="nav-section-label">Main Menu</div>}

        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={onCloseMobile}
              title={isCollapsed ? item.label : undefined}
            >
              <Icon size={18} className="nav-icon" />
              <span className="nav-text">{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Sidebar Footer */}
      <div className="sidebar-footer">
        {/* User Profile Widget */}
        {user && !isCollapsed && (
          <div className="user-profile-widget">
            <div className="user-profile-info">
              <div className="user-avatar-circle">
                {user.avatar_url ? (
                  <img src={user.avatar_url} alt={user.full_name} />
                ) : (
                  <span>{initials}</span>
                )}
              </div>
              <div className="user-profile-text">
                <span className="user-profile-name" title={user.full_name}>
                  {user.full_name}
                </span>
                <span className="user-profile-email" title={user.email}>
                  {user.email}
                </span>
              </div>
            </div>
            <button
              className="btn-logout"
              onClick={logout}
              title="Sign out of DocuMind AI"
              aria-label="Logout"
            >
              <LogOut size={16} />
            </button>
          </div>
        )}

        {/* Backend health status indicator */}

        <div
          className="backend-status-pill"
          title={`Backend status: ${backendStatus.toUpperCase()} • ${
            backendInfo?.uptime_message || 'Connecting to FastAPI...'
          }`}
        >
          <span className={`status-dot ${backendStatus}`} />
          <span>
            {backendStatus === 'online'
              ? 'FastAPI Connected'
              : backendStatus === 'checking'
              ? 'Connecting...'
              : 'Backend Offline'}
          </span>
        </div>

        {/* Configurable GitHub Link */}
        {githubRepoUrl ? (
          <a
            href={githubRepoUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="sidebar-github-link"
            title="GitHub Repository"
          >
            <Github size={18} />
            <span>GitHub Repository</span>
          </a>
        ) : null}
      </div>
    </aside>
  );
};

export default Sidebar;
