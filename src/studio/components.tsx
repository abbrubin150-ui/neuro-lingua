/** Small presentational building blocks of the Studio. */
import { useEffect, useId, useState, type ReactNode } from 'react';

export function NumberField(props: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number | 'any';
  hint?: string;
  disabled?: boolean;
}) {
  const { label, value, onChange, min, max, step = 'any', hint, disabled } = props;
  const [draft, setDraft] = useState(String(value));
  // Follow outside changes (presets, reset) without fighting the user's typing ("0.00" → 0).
  useEffect(() => {
    setDraft((d) => (d.trim() !== '' && Number(d) === value ? d : String(value)));
  }, [value]);
  const invalid =
    draft.trim() === '' ||
    !Number.isFinite(Number(draft)) ||
    (min !== undefined && Number(draft) < min) ||
    (max !== undefined && Number(draft) > max);
  return (
    <label className={`field${invalid ? ' field--invalid' : ''}`}>
      <span className="field__label">{label}</span>
      <input
        type="number"
        inputMode="decimal"
        value={draft}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => {
          const text = e.target.value;
          setDraft(text);
          const n = Number(text);
          if (text.trim() !== '' && Number.isFinite(n)) onChange(n);
        }}
      />
      {hint && <span className="field__hint">{hint}</span>}
    </label>
  );
}

export function SelectField<T extends string>(props: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <label className="field">
      <span className="field__label">{props.label}</span>
      <select
        value={props.value}
        disabled={props.disabled}
        onChange={(e) => props.onChange(e.target.value as T)}
      >
        {props.options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Toggle(props: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="toggle">
      <input
        type="checkbox"
        checked={props.checked}
        disabled={props.disabled}
        onChange={(e) => props.onChange(e.target.checked)}
      />
      <span className="toggle__track" aria-hidden="true" />
      <span>{props.label}</span>
    </label>
  );
}

export function Segmented<T extends string>(props: {
  label: string;
  value: T;
  options: { value: T; label: string; note?: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="segmented-field">
      <span className="field__label" id={id}>
        {props.label}
      </span>
      <div className="segmented" role="group" aria-labelledby={id}>
        {props.options.map((o) => (
          <button
            key={o.value}
            type="button"
            className="segmented__option"
            aria-pressed={props.value === o.value}
            disabled={props.disabled}
            onClick={() => props.onChange(o.value)}
          >
            <span className="segmented__title">{o.label}</span>
            {o.note && <span className="segmented__note">{o.note}</span>}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Card(props: {
  title?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${props.className ?? ''}`}>
      {(props.title || props.actions) && (
        <header className="card__head">
          {props.title && <h2 className="card__title">{props.title}</h2>}
          {props.actions && <div className="card__actions">{props.actions}</div>}
        </header>
      )}
      {props.children}
    </section>
  );
}

export function Stat(props: { label: string; value: string; tone?: 'accent' | 'muted' }) {
  return (
    <div className={`stat${props.tone ? ` stat--${props.tone}` : ''}`}>
      <span className="stat__label">{props.label}</span>
      <span className="stat__value">{props.value}</span>
    </div>
  );
}

/** Line chart of one series over epochs (inline SVG, no dependencies). */
export function LineChart(props: {
  label: string;
  points: { x: number; y: number }[];
  color: 'a' | 'b';
  digits?: number;
}) {
  const { points, digits = 3 } = props;
  const W = 320;
  const H = 140;
  const pad = { l: 40, r: 10, t: 12, b: 22 };
  if (points.length === 0) return null;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs, xMin + 1);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMax - yMin < 1e-9) {
    yMin -= 0.5;
    yMax += 0.5;
  }
  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin)) * (W - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (y - yMin) / (yMax - yMin)) * (H - pad.t - pad.b);
  const d = points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`);
  const lastPoint = points[points.length - 1];
  const best = points.reduce((a, b) => (b.y < a.y ? b : a));
  return (
    <figure className={`chart chart--${props.color}`}>
      <figcaption className="chart__caption">
        <span>{props.label}</span>
        <span className="chart__value">{lastPoint.y.toFixed(digits)}</span>
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={props.label}
        style={{ direction: 'ltr' }}
      >
        {[0, 0.5, 1].map((f) => {
          const y = pad.t + f * (H - pad.t - pad.b);
          const v = yMax - f * (yMax - yMin);
          return (
            <g key={f}>
              <line className="chart__grid" x1={pad.l} x2={W - pad.r} y1={y} y2={y} />
              <text className="chart__tick" x={pad.l - 6} y={y + 3} textAnchor="end">
                {v.toFixed(2)}
              </text>
            </g>
          );
        })}
        <text className="chart__tick" x={pad.l} y={H - 6}>
          {xMin}
        </text>
        <text className="chart__tick" x={W - pad.r} y={H - 6} textAnchor="end">
          {xMax}
        </text>
        <path className="chart__line" d={d.join(' ')} />
        {points.map((p) => (
          <circle
            key={p.x}
            className={p === best ? 'chart__dot chart__dot--best' : 'chart__dot'}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={p === best ? 3.5 : 2}
          />
        ))}
      </svg>
    </figure>
  );
}

export function ProgressBar(props: { value: number; label: string }) {
  const pct = Math.max(0, Math.min(100, props.value * 100));
  return (
    <div
      className="progress"
      role="progressbar"
      aria-label={props.label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
    >
      <div className="progress__fill" style={{ width: `${pct}%` }} />
    </div>
  );
}
