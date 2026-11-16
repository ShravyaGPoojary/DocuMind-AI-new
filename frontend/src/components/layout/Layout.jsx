import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';
import Toast from '../common/Toast';
import UploadModal from '../common/UploadModal';
import DocumentDetailsModal from '../common/DocumentDetailsModal';
import { useDocuments } from '../../context/DocumentContext';

const Layout = () => {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const {
    isUploadModalOpen,
    setIsUploadModalOpen,
    selectedDocForDetails,
    setSelectedDocForDetails,
  } = useDocuments();

  return (
    <div className="app-container">
      {/* Collapsible Sidebar */}
      <Sidebar
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        isMobileOpen={isMobileMenuOpen}
        onCloseMobile={() => setIsMobileMenuOpen(false)}
      />

      {/* Main Content Wrapper */}
      <div className={`main-wrapper ${isSidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
        <Header onOpenMobileMenu={() => setIsMobileMenuOpen(true)} />

        <main className="page-content">
          <Outlet />
        </main>
      </div>

      {/* Global Modals & Notifications */}
      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
      />

      <DocumentDetailsModal
        document={selectedDocForDetails}
        isOpen={!!selectedDocForDetails}
        onClose={() => setSelectedDocForDetails(null)}
      />

      <Toast />
    </div>
  );
};

export default Layout;
