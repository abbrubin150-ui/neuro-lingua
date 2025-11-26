import React from 'react';
import ReactDOM from 'react-dom/client';
import NeuroLinguaDomesticaV324 from './App';
import { ProjectProvider } from './contexts/ProjectContext';
import { BrainProvider } from './contexts/BrainContext';

// Add loading timeout to prevent stuck loading screen
const LOADING_TIMEOUT = 10000; // 10 seconds

// Show error if app doesn't load within timeout
const timeoutId = setTimeout(() => {
  const root = document.getElementById('root');
  if (root && root.innerHTML.includes('Loading Neuro-Lingua')) {
    root.innerHTML = `
      <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
        font-family: system-ui, sans-serif;
        padding: 20px;
      ">
        <div style="text-align: center; max-width: 600px;">
          <h1 style="font-size: 2rem; margin-bottom: 1rem; color: #ef4444;">⚠️ Loading Error</h1>
          <p style="color: #94a3b8; margin-bottom: 1rem;">
            The application failed to load. This could be due to:
          </p>
          <ul style="color: #94a3b8; text-align: left; margin-bottom: 1rem;">
            <li>Browser compatibility issues (try Chrome or Edge)</li>
            <li>JavaScript errors preventing initialization</li>
            <li>Network connectivity problems</li>
          </ul>
          <button
            onclick="location.reload()"
            style="
              padding: 12px 24px;
              background: #6366f1;
              border: none;
              border-radius: 8px;
              color: white;
              font-weight: 600;
              cursor: pointer;
              font-size: 1rem;
            "
          >
            🔄 Reload Page
          </button>
          <p style="color: #94a3b8; margin-top: 1rem; font-size: 0.875rem;">
            If the problem persists, check the browser console for errors (F12).
          </p>
        </div>
      </div>
    `;
    console.error('App loading timeout - check for initialization errors');
  }
}, LOADING_TIMEOUT);

try {
  const rootElement = document.getElementById('root');
  if (!rootElement) {
    throw new Error('Root element not found');
  }

  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <BrainProvider>
        <ProjectProvider>
          <NeuroLinguaDomesticaV324 />
        </ProjectProvider>
      </BrainProvider>
    </React.StrictMode>
  );

  // Clear timeout if app loads successfully
  clearTimeout(timeoutId);
} catch (error) {
  console.error('Failed to initialize React app:', error);
  const root = document.getElementById('root');
  if (root) {
    root.innerHTML = `
      <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
        font-family: system-ui, sans-serif;
        padding: 20px;
      ">
        <div style="text-align: center; max-width: 600px;">
          <h1 style="font-size: 2rem; margin-bottom: 1rem; color: #ef4444;">❌ Initialization Error</h1>
          <p style="color: #94a3b8; margin-bottom: 1rem;">
            Failed to initialize the application: ${error instanceof Error ? error.message : 'Unknown error'}
          </p>
          <button
            onclick="location.reload()"
            style="
              padding: 12px 24px;
              background: #6366f1;
              border: none;
              border-radius: 8px;
              color: white;
              font-weight: 600;
              cursor: pointer;
              font-size: 1rem;
            "
          >
            🔄 Reload Page
          </button>
        </div>
      </div>
    `;
  }
}
