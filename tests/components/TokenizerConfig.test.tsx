import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { TokenizerConfig } from '../../src/components/TokenizerConfig';
import { DEFAULT_CUSTOM_TOKENIZER_PATTERN } from '../../src/config/constants';

describe('TokenizerConfig', () => {
  it('switches between tokenizer modes', () => {
    const props = createProps();
    render(<TokenizerConfig {...props} />);
    const select = screen.getByLabelText(/Tokenizer mode/i);

    fireEvent.change(select, { target: { value: 'ascii' } });
    expect(props.onConfigChange).toHaveBeenCalledWith({ mode: 'ascii' });

    fireEvent.change(select, { target: { value: 'custom' } });
    expect(props.onConfigChange).toHaveBeenCalledWith({
      mode: 'custom',
      pattern: DEFAULT_CUSTOM_TOKENIZER_PATTERN
    });
  });

  it('validates custom patterns', () => {
    const props = createProps({
      config: { mode: 'custom', pattern: '[a-z]+' },
      customPattern: '[a-z]+'
    });
    render(<TokenizerConfig {...props} />);

    const input = screen.getByLabelText(/Custom tokenizer pattern/i);

    fireEvent.change(input, { target: { value: '' } });
    expect(props.onError).toHaveBeenCalledWith(
      'Enter a regular expression pattern to enable custom tokenization.'
    );

    fireEvent.change(input, { target: { value: '[A-Z]+' } });
    expect(props.onConfigChange).toHaveBeenCalledWith({ mode: 'custom', pattern: '[A-Z]+' });
  });
});

function createProps(
  overrides: Partial<Parameters<typeof TokenizerConfig>[0]> = {}
): Parameters<typeof TokenizerConfig>[0] {
  return {
    config: { mode: 'unicode', ...(overrides.config ?? {}) },
    customPattern: overrides.customPattern ?? '',
    error: overrides.error ?? null,
    onConfigChange: overrides.onConfigChange ?? vi.fn(),
    onCustomPatternChange: overrides.onCustomPatternChange ?? vi.fn(),
    onError: overrides.onError ?? vi.fn(),
    onMessage: overrides.onMessage ?? vi.fn()
  };
}
