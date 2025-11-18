import React from 'react';
import ReactDOM from 'react-dom/client';
import NeuroLinguaDomesticaV32 from './App';
import { ProjectProvider } from './contexts/ProjectContext';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <ProjectProvider>
      <NeuroLinguaDomesticaV32 />
    </ProjectProvider>
  </React.StrictMode>
);
