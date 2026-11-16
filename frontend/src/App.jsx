import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import DashboardPage from './pages/DashboardPage';
import DocumentsPage from './pages/DocumentsPage';
import AIChatPage from './pages/AIChatPage';
import SummarizationPage from './pages/SummarizationPage';
import MultiDocQAPage from './pages/MultiDocQAPage';
import DocumentComparisonPage from './pages/DocumentComparisonPage';
import DocumentIntelligencePage from './pages/DocumentIntelligencePage';
import SettingsPage from './pages/SettingsPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ProtectedRoute from './components/auth/ProtectedRoute';
import { AuthProvider } from './context/AuthContext';
import { DocumentProvider } from './context/DocumentContext';

function App() {
  return (
    <AuthProvider>
      <DocumentProvider>
        <Routes>
          {/* Public Authentication Routes */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* Protected Application Workspace */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<DashboardPage />} />
            <Route path="documents" element={<DocumentsPage />} />
            <Route path="chat" element={<AIChatPage />} />
            <Route path="intelligence" element={<DocumentIntelligencePage />} />
            <Route path="summarize" element={<SummarizationPage />} />
            <Route path="multi-doc" element={<MultiDocQAPage />} />
            <Route path="compare" element={<DocumentComparisonPage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </DocumentProvider>
    </AuthProvider>
  );
}

export default App;

