import type { CerebroBubble, InjectionEvent, InjectionProposal } from '../types/injection';
import type { InjectableLayer, InjectionSnapshot } from '../lib/expandable/InjectableLayer';
import { cloneWeights } from '../lib/expandable/InjectableLayer';
import { InjectionEngine } from '../lib/expandable/InjectionEngine';

export interface InjectionLedgerAdapter {
  appendEvent: (event: InjectionEvent) => void;
  undoLast?: (event: InjectionEvent) => void;
}

export class InjectionRunSession {
  private readonly snapshots: InjectionSnapshot[] = [];

  private readonly events: InjectionEvent[] = [];

  constructor(
    private readonly layer: InjectableLayer,
    private readonly ledger?: InjectionLedgerAdapter,
    private readonly engine: InjectionEngine = new InjectionEngine()
  ) {}

  get history(): InjectionEvent[] {
    return this.events;
  }

  captureSnapshot(): void {
    const weights = cloneWeights(this.layer.exportWeights());
    this.snapshots.push({
      weights,
      capturedAt: Date.now()
    });
  }

  propose(bubbles: CerebroBubble[]): InjectionProposal {
    const diagnostics = this.engine.diagnose(bubbles, this.layer);
    return this.engine.propose(diagnostics, this.layer.getTarget());
  }

  inject(proposal: InjectionProposal, bubbles: CerebroBubble[]): InjectionEvent {
    this.captureSnapshot();
    const event = this.engine.execute(proposal, this.layer, bubbles);

    if (!event.accepted) {
      // keep the snapshot for debugging but do not clear
      this.events.push(event);
      this.ledger?.appendEvent(event);
      return event;
    }

    // accepted path
    this.events.push(event);
    this.ledger?.appendEvent(event);
    return event;
  }

  undoLast(): InjectionEvent | null {
    const lastSnapshot = this.snapshots.pop();
    const lastEvent = this.events.pop();

    if (lastSnapshot) {
      this.layer.importWeights(lastSnapshot.weights);
    }

    if (lastEvent) {
      this.ledger?.undoLast?.(lastEvent);
      return lastEvent;
    }

    return null;
  }
}

export function createLedgerAdapter(records: InjectionEvent[]): InjectionLedgerAdapter {
  return {
    appendEvent: (event) => {
      records.push(event);
    },
    undoLast: (event) => {
      const idx = records.findIndex((e) => e.seed === event.seed);
      if (idx >= 0) {
        records.splice(idx, 1);
      }
    }
  };
}
