import React from 'react';
import ReactDOM from 'react-dom/client';
import StudioApp from './StudioApp';
import { ErrorBoundary } from '../components/ErrorBoundary';
import { registerOfflineSupport } from '../pwa/register';
import './studio.css';

const rootElement = document.getElementById('root');
if (!rootElement) throw new Error('Root element not found');

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <ErrorBoundary>
      <StudioApp />
    </ErrorBoundary>
  </React.StrictMode>
);

registerOfflineSupport();
