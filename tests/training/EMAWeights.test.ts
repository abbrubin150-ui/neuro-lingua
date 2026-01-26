import { describe, expect, it, beforeEach } from 'vitest';
import { EMATracker, type WeightSnapshot } from '../../src/training/EMAWeights';

function makeWeights(fill: number): WeightSnapshot {
  return {
    embedding: [[fill, fill], [fill, fill]],
    wHidden: [[fill, fill], [fill, fill]],
    wOutput: [[fill, fill], [fill, fill]],
    bHidden: [fill, fill],
    bOutput: [fill, fill]
  };
}

describe('EMATracker', () => {
  let tracker: EMATracker;

  beforeEach(() => {
    tracker = new EMATracker(0.9);
  });

  it('initializes in non-initialized state', () => {
    expect(tracker.isInitialized()).toBe(false);
    expect(tracker.getShadowWeights()).toBeNull();
    expect(tracker.getNumUpdates()).toBe(0);
  });

  it('first update copies current weights', () => {
    const weights = makeWeights(1.0);
    tracker.update(weights);

    expect(tracker.isInitialized()).toBe(true);
    expect(tracker.getNumUpdates()).toBe(1);

    const shadow = tracker.getShadowWeights()!;
    expect(shadow.bHidden[0]).toBeCloseTo(1.0);
    expect(shadow.embedding[0][0]).toBeCloseTo(1.0);
  });

  it('subsequent updates apply EMA formula', () => {
    // decay = 0.9
    // After first update with 1.0: shadow = 1.0
    // After second update with 2.0: shadow = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    tracker.update(makeWeights(1.0));
    tracker.update(makeWeights(2.0));

    expect(tracker.getNumUpdates()).toBe(2);

    const shadow = tracker.getShadowWeights()!;
    expect(shadow.bHidden[0]).toBeCloseTo(1.1);
    expect(shadow.embedding[0][0]).toBeCloseTo(1.1);
    expect(shadow.wHidden[0][0]).toBeCloseTo(1.1);
    expect(shadow.wOutput[0][0]).toBeCloseTo(1.1);
    expect(shadow.bOutput[0]).toBeCloseTo(1.1);
  });

  it('converges toward constant input', () => {
    tracker = new EMATracker(0.9);
    tracker.update(makeWeights(0.0));

    // Feed constant 1.0 many times
    for (let i = 0; i < 100; i++) {
      tracker.update(makeWeights(1.0));
    }

    const shadow = tracker.getShadowWeights()!;
    // After many updates, shadow should converge close to 1.0
    expect(shadow.bHidden[0]).toBeCloseTo(1.0, 2);
  });

  it('higher decay means slower adaptation', () => {
    const slowTracker = new EMATracker(0.999);
    const fastTracker = new EMATracker(0.9);

    slowTracker.update(makeWeights(0.0));
    fastTracker.update(makeWeights(0.0));

    // One update with 10.0
    slowTracker.update(makeWeights(10.0));
    fastTracker.update(makeWeights(10.0));

    const slowShadow = slowTracker.getShadowWeights()!;
    const fastShadow = fastTracker.getShadowWeights()!;

    // Fast tracker should be closer to 10.0
    expect(fastShadow.bHidden[0]).toBeGreaterThan(slowShadow.bHidden[0]);
  });

  it('returns deep copies from getShadowWeights', () => {
    tracker.update(makeWeights(1.0));
    const shadow1 = tracker.getShadowWeights()!;
    const shadow2 = tracker.getShadowWeights()!;

    // Mutating one copy should not affect the other
    shadow1.bHidden[0] = 999;
    expect(shadow2.bHidden[0]).toBeCloseTo(1.0);
  });

  it('does not mutate input weights', () => {
    const weights = makeWeights(5.0);
    tracker.update(weights);
    tracker.update(makeWeights(0.0));

    // Original weights should be unchanged
    expect(weights.bHidden[0]).toBeCloseTo(5.0);
    expect(weights.embedding[0][0]).toBeCloseTo(5.0);
  });

  it('get/set decay works', () => {
    expect(tracker.getDecay()).toBe(0.9);
    tracker.setDecay(0.99);
    expect(tracker.getDecay()).toBe(0.99);
  });

  it('clamps decay to [0, 1]', () => {
    tracker.setDecay(-0.5);
    expect(tracker.getDecay()).toBe(0);
    tracker.setDecay(1.5);
    expect(tracker.getDecay()).toBe(1);
  });

  it('reset clears state', () => {
    tracker.update(makeWeights(1.0));
    expect(tracker.isInitialized()).toBe(true);

    tracker.reset();
    expect(tracker.isInitialized()).toBe(false);
    expect(tracker.getShadowWeights()).toBeNull();
    expect(tracker.getNumUpdates()).toBe(0);
  });

  it('export/import preserves state', () => {
    tracker.update(makeWeights(1.0));
    tracker.update(makeWeights(2.0));

    const exported = tracker.exportState();
    expect(exported.decay).toBe(0.9);
    expect(exported.numUpdates).toBe(2);
    expect(exported.shadowWeights).not.toBeNull();

    // Import into a new tracker
    const newTracker = new EMATracker(0.5);
    newTracker.importState(exported);

    expect(newTracker.getDecay()).toBe(0.9);
    expect(newTracker.getNumUpdates()).toBe(2);

    const shadow1 = tracker.getShadowWeights()!;
    const shadow2 = newTracker.getShadowWeights()!;
    expect(shadow2.bHidden[0]).toBeCloseTo(shadow1.bHidden[0]);
  });

  it('exported state is a deep copy', () => {
    tracker.update(makeWeights(1.0));
    const exported = tracker.exportState();

    // Mutate the export
    exported.shadowWeights!.bHidden[0] = 999;

    // Tracker should be unaffected
    const shadow = tracker.getShadowWeights()!;
    expect(shadow.bHidden[0]).toBeCloseTo(1.0);
  });

  it('handles multiple sequential updates correctly', () => {
    // decay = 0.9
    // u1: shadow = 1.0
    // u2: shadow = 0.9*1.0 + 0.1*3.0 = 1.2
    // u3: shadow = 0.9*1.2 + 0.1*5.0 = 1.58
    tracker.update(makeWeights(1.0));
    tracker.update(makeWeights(3.0));
    tracker.update(makeWeights(5.0));

    const shadow = tracker.getShadowWeights()!;
    expect(shadow.bHidden[0]).toBeCloseTo(1.58);
  });
});
