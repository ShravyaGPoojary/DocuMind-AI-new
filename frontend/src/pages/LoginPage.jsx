import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Sparkles, LogIn, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import '../styles/auth.css';

const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [localError, setLocalError] = useState(null);

  const { login, loginWithGoogle, isAuthenticated, authError } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const googleBtnRef = useRef(null);

  const googleClientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
  const from = location.state?.from?.pathname || '/';

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, from]);

  // Initialize Google Identity Services if client ID is configured
  useEffect(() => {
    if (!googleClientId) return;

    const initializeGsi = () => {
      if (window.google?.accounts?.id && googleBtnRef.current) {
        window.google.accounts.id.initialize({
          client_id: googleClientId,
          callback: async (response) => {
            if (response.credential) {
              try {
                setIsSubmitting(true);
                setLocalError(null);
                await loginWithGoogle(response.credential);
                navigate(from, { replace: true });
              } catch (err) {
                setLocalError(err.message || 'Google Sign-In failed');
              } finally {
                setIsSubmitting(false);
              }
            }
          },
        });

        window.google.accounts.id.renderButton(googleBtnRef.current, {
          theme: 'outline',
          size: 'large',
          width: 380,
          text: 'signin_with',
          shape: 'rectangular',
        });
      }
    };

    // Check if script is already present
    if (window.google?.accounts?.id) {
      initializeGsi();
    } else {
      const script = document.createElement('script');
      script.src = 'https://accounts.google.com/gsi/client';
      script.async = true;
      script.defer = true;
      script.onload = initializeGsi;
      document.body.appendChild(script);
    }
  }, [googleClientId, loginWithGoogle, navigate, from]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email.trim() || !password) {
      setLocalError('Please enter both email and password.');
      return;
    }

    setIsSubmitting(true);
    setLocalError(null);

    try {
      await login(email.trim(), password);
      navigate(from, { replace: true });
    } catch (err) {
      setLocalError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const autofillDemo = () => {
    setEmail('admin@documind.ai');
    setPassword('DocuMind@2026');
    setLocalError(null);
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        {/* Brand Header */}
        <div className="auth-header">
          <div className="auth-logo-box">
            <Sparkles size={28} />
          </div>
          <h1>Welcome Back</h1>
          <p>Sign in to access your documents and intelligence workbench</p>
        </div>

        {/* Error Notification */}
        {(localError || authError) && (
          <div className="auth-alert-error">
            <AlertCircle size={16} />
            <span>{localError || authError}</span>
          </div>
        )}

        {/* Login Form */}
        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="auth-field">
            <label htmlFor="loginEmail">Email Address</label>
            <input
              id="loginEmail"
              type="email"
              placeholder="you@domain.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={isSubmitting}
              autoComplete="email"
            />
          </div>

          <div className="auth-field">
            <label htmlFor="loginPassword">Password</label>
            <input
              id="loginPassword"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={isSubmitting}
              autoComplete="current-password"
            />
          </div>

          <button type="submit" className="btn-auth-submit" disabled={isSubmitting}>
            <LogIn size={18} />
            {isSubmitting ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        {/* Google OAuth Section */}
        <div className="auth-divider">or continue with</div>

        <div className="google-auth-wrapper">
          {googleClientId ? (
            <div ref={googleBtnRef} style={{ width: '100%', display: 'flex', justifyContent: 'center' }} />
          ) : (
            <div
              style={{
                fontSize: '0.8rem',
                color: 'var(--text-tertiary)',
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-sm)',
                padding: '8px 12px',
                textAlign: 'center',
                width: '100%',
              }}
            >
              Google Sign-In ready (Set <code>VITE_GOOGLE_CLIENT_ID</code> in <code>.env</code> to activate)
            </div>
          )}
        </div>

        {/* Demo Credentials Helper */}
        <div className="demo-credentials-box">
          <div>
            <strong>Quick Demo Account:</strong> <code>admin@documind.ai</code> / <code>DocuMind@2026</code>
          </div>
          <button type="button" className="btn-demo-autofill" onClick={autofillDemo}>
            Click to auto-fill demo credentials
          </button>
        </div>

        {/* Footer Link */}
        <div className="auth-footer">
          Don't have an account yet?
          <Link to="/register" className="auth-link">
            Create an account
          </Link>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;

