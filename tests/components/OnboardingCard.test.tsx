import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { OnboardingCard } from '../../src/components/OnboardingCard';
import { STORAGE_KEYS } from '../../src/config/constants';

const strings = {
  welcomeTitle: 'Welcome to Neuro-Lingua',
  privacyWarningTitle: 'Privacy reminder',
  privacyWarningLead: 'Everything stays local.',
  privacyWarningBody: 'Nothing leaves your browser.',
  bulletPauseResume: 'Pause and resume whenever you like.',
  bulletImportExport: 'Import and export artifacts.',
  bulletPersistence: 'Sessions persist automatically.',
  gotIt: 'Got it',
  reopenInfo: 'You can reopen this card from the help menu.'
};

describe('OnboardingCard', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('does not render when show is false', () => {
    const { container } = render(
      <OnboardingCard show={false} onDismiss={vi.fn()} strings={strings} direction="ltr" />
    );

    expect(container).toBeEmptyDOMElement();
  });

  it('renders copy and persists dismissal to localStorage', () => {
    const onDismiss = vi.fn();
    render(<OnboardingCard show onDismiss={onDismiss} strings={strings} direction="ltr" />);

    expect(screen.getByText(strings.welcomeTitle)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: strings.gotIt }));

    expect(localStorage.getItem(STORAGE_KEYS.ONBOARDING_DISMISSED)).toBe('true');
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });
});
