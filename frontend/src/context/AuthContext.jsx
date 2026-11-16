import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import {
  loginUser,
  registerUser,
  googleAuth,
  getCurrentUser,
  logoutUser,
} from '../services/api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    try {
      const savedUser = localStorage.getItem('documind_user');
      return savedUser ? JSON.parse(savedUser) : null;
    } catch {
      return null;
    }
  });
  const [token, setToken] = useState(() => localStorage.getItem('documind_token'));
  const [isLoading, setIsLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  // Validate existing token on mount
  useEffect(() => {
    const verifyToken = async () => {
      const storedToken = localStorage.getItem('documind_token');
      if (!storedToken) {
        setIsLoading(false);
        return;
      }
      try {
        const userData = await getCurrentUser();
        setUser(userData);
        localStorage.setItem('documind_user', JSON.stringify(userData));
      } catch (err) {
        // Token expired or invalid
        localStorage.removeItem('documind_token');
        localStorage.removeItem('documind_user');
        setToken(null);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    verifyToken();
  }, []);

  const login = async (email, password) => {
    setAuthError(null);
    try {
      const data = await loginUser(email, password);
      localStorage.setItem('documind_token', data.access_token);
      localStorage.setItem('documind_user', JSON.stringify(data.user));
      setToken(data.access_token);
      setUser(data.user);
      return data.user;
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Login failed. Please check your credentials.';
      setAuthError(msg);
      throw new Error(msg);
    }
  };

  const register = async (fullName, email, password) => {
    setAuthError(null);
    try {
      const data = await registerUser(fullName, email, password);
      localStorage.setItem('documind_token', data.access_token);
      localStorage.setItem('documind_user', JSON.stringify(data.user));
      setToken(data.access_token);
      setUser(data.user);
      return data.user;
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Registration failed.';
      setAuthError(msg);
      throw new Error(msg);
    }
  };

  const loginWithGoogle = async (credential) => {
    setAuthError(null);
    try {
      const data = await googleAuth(credential);
      localStorage.setItem('documind_token', data.access_token);
      localStorage.setItem('documind_user', JSON.stringify(data.user));
      setToken(data.access_token);
      setUser(data.user);
      return data.user;
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Google authentication failed.';
      setAuthError(msg);
      throw new Error(msg);
    }
  };

  const logout = useCallback(async () => {
    try {
      await logoutUser();
    } catch {
      // Ignore network errors on logout
    } finally {
      localStorage.removeItem('documind_token');
      localStorage.removeItem('documind_user');
      setToken(null);
      setUser(null);
      setAuthError(null);
    }
  }, []);

  const value = {
    user,
    token,
    isAuthenticated: !!token && !!user,
    isLoading,
    authError,
    setAuthError,
    login,
    register,
    loginWithGoogle,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;

