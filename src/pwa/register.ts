/**
 * Register the offline service worker of the built app (see src/pwa/serviceWorker.ts).
 *
 * Only in production builds served over http(s): the development server has no worker, and the
 * desktop app loads its files from disk (file://), where service workers do not apply and are not
 * needed. Failure to register only means the app needs the network to start.
 */
import { SERVICE_WORKER_FILE } from './serviceWorker';

export function registerOfflineSupport(): void {
  if (!import.meta.env.PROD) return;
  if (typeof window === 'undefined' || typeof navigator === 'undefined') return;
  if (!('serviceWorker' in navigator)) return;
  if (location.protocol !== 'https:' && location.protocol !== 'http:') return;
  if (!window.isSecureContext) return;
  const register = () => {
    navigator.serviceWorker
      .register(`./${SERVICE_WORKER_FILE}`)
      .catch((error: unknown) => console.warn('Offline support is not available:', error));
  };
  if (document.readyState === 'complete') register();
  else window.addEventListener('load', register, { once: true });
}
