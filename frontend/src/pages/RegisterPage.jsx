import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Sparkles, UserPlus, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import '../styles/auth.css';

const RegisterPage = () => {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [localError, setLocalError] = useState(null);

  const { register, loginWithGoogle, isAuthenticated, authError } = useAuth();
  const navigate = useNavigate();
  const googleBtnRef = useRef(null);

  const googleClientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  // Initialize Google Identity Services
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
                navigate('/', { replace: true });
              } catch (err) {
                setLocalError(err.message || 'Google registration failed');
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
          text: 'signup_with',
          shape: 'rectangular',
        });
      }
    };

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
  }, [googleClientId, loginWithGoogle, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLocalError(null);

    if (!fullName.trim() || !email.trim() || !password) {
      setLocalError('Please fill in all required fields.');
      return;
    }

    if (password.length < 6) {
      setLocalError('Password must be at least 6 characters long.');
      return;
    }

    if (password !== confirmPassword) {
      setLocalError('Passwords do not match.');
      return;
    }

    setIsSubmitting(true);

    try {
      await register(fullName.trim(), email.trim(), password);
      navigate('/', { replace: true });
    } catch (err) {
      setLocalError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        {/* Brand Header */}
        <div className="auth-header">
          <div className="auth-logo-box">
            <Sparkles size={28} />
          </div>
          <h1>Create an Account</h1>
          <p>Get started with private, isolated AI document intelligence</p>
        </div>

        {/* Error Notification */}
        {(localError || authError) && (
          <div className="auth-alert-error">
            <AlertCircle size={16} />
            <span>{localError || authError}</span>
          </div>
        )}

        {/* Registration Form */}
        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="auth-field">
            <label htmlFor="regName">Full Name</label>
            <input
              id="regName"
              type="text"
              placeholder="e.g. Jane Doe"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              required
              disabled={isSubmitting}
              autoComplete="name"
            />
          </div>

          <div className="auth-field">
            <label htmlFor="regEmail">Email Address</label>
            <input
              id="regEmail"
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
            <label htmlFor="regPassword">Password (min 6 characters)</label>
            <input
              id="regPassword"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={isSubmitting}
              autoComplete="new-password"
            />
          </div>

          <div className="auth-field">
            <label htmlFor="regConfirmPassword">Confirm Password</label>
            <input
              id="regConfirmPassword"
              type="password"
              placeholder="••••••••"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              disabled={isSubmitting}
              autoComplete="new-password"
            />
          </div>

          <button type="submit" className="btn-auth-submit" disabled={isSubmitting}>
            <UserPlus size={18} />
            {isSubmitting ? 'Creating account...' : 'Create Account'}
          </button>
        </form>

        {/* Google OAuth Section */}
        <div className="auth-divider">or sign up with</div>

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
              Google Sign-Up ready (Set <code>VITE_GOOGLE_CLIENT_ID</code> in <code>.env</code> to activate)
            </div>
          )}
        </div>

        {/* Footer Link */}
        <div className="auth-footer">
          Already have an account?
          <Link to="/login" className="auth-link">
            Sign In
          </Link>
        </div>
      </div>
    </div>
  );
};

export default RegisterPage;

