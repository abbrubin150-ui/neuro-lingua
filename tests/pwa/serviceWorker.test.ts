// @vitest-environment node
import { describe, expect, it, vi } from 'vitest';
import {
  CACHE_PREFIX,
  buildServiceWorkerSource,
  buildVersion,
  selectPrecacheFiles
} from '../../src/pwa/serviceWorker';

const SW_URL = 'https://example.test/app/sw.js';
const at = (path: string) => new URL(path, SW_URL).href;

/** Minimal stand-in for a same-origin ("basic") network response. */
class FakeResponse {
  readonly ok: boolean;
  readonly type = 'basic';
  readonly headers = new Headers();
  readonly statusText = '';
  constructor(
    readonly body: string,
    readonly status = 200,
    readonly redirected = false
  ) {
    this.ok = status >= 200 && status < 300;
  }
  clone() {
    return new FakeResponse(this.body, this.status, this.redirected);
  }
  async text() {
    return this.body;
  }
}

type RequestLike = { url: string; method: string; mode: string };
const request = (url: string, init: Partial<RequestLike> = {}): RequestLike => ({
  url,
  method: 'GET',
  mode: 'no-cors',
  ...init
});

function createWorker(precache: string[], version = 'v2') {
  const listeners = new Map<string, (event: unknown) => void>();
  const network = new Map<string, () => FakeResponse>();
  const fetched: string[] = [];
  let online = true;

  const fetch = async (input: RequestLike | string) => {
    const url = typeof input === 'string' ? at(input) : input.url;
    fetched.push(url);
    if (!online) throw new TypeError('Failed to fetch');
    return network.get(url)?.() ?? new FakeResponse('not found', 404);
  };

  class FakeCache {
    entries = new Map<string, FakeResponse>();
    async addAll(urls: string[]) {
      for (const url of urls) {
        const response = await fetch(url);
        if (!response.ok) throw new TypeError(`Request failed: ${url}`);
        this.entries.set(at(url), response);
      }
    }
    async put(req: RequestLike | string, response: FakeResponse) {
      this.entries.set(typeof req === 'string' ? at(req) : req.url, response);
    }
    async match(req: RequestLike | string) {
      return this.entries.get(typeof req === 'string' ? at(req) : req.url)?.clone();
    }
  }

  const stores = new Map<string, FakeCache>();
  const caches = {
    open: async (name: string) => {
      if (!stores.has(name)) stores.set(name, new FakeCache());
      return stores.get(name)!;
    },
    keys: async () => [...stores.keys()],
    delete: async (name: string) => stores.delete(name)
  };
  const self = {
    location: new URL(SW_URL),
    addEventListener: (type: string, listener: (event: unknown) => void) =>
      listeners.set(type, listener),
    skipWaiting: vi.fn(async () => undefined),
    clients: { claim: vi.fn(async () => undefined) }
  };

  const source = buildServiceWorkerSource({ version, precache });
  new Function('self', 'caches', 'fetch', source)(self, caches, fetch);

  return {
    self,
    stores,
    network,
    fetched,
    setOnline: (value: boolean) => {
      online = value;
    },
    async lifecycle(type: 'install' | 'activate') {
      let pending: Promise<unknown> = Promise.resolve();
      listeners.get(type)!({ waitUntil: (p: Promise<unknown>) => (pending = p) });
      await pending;
    },
    /** Dispatch a fetch event; returns undefined when the worker does not answer it. */
    async fetchEvent(req: RequestLike): Promise<FakeResponse | Response | undefined> {
      let answer: Promise<FakeResponse | Response> | undefined;
      listeners.get('fetch')!({ request: req, respondWith: (p: Promise<FakeResponse>) => (answer = p) });
      return answer ? await answer : undefined;
    }
  };
}

const BUILD = ['assets/index-abc.js', 'assets/chunk-def.js', 'index.html', 'icon.svg'];

function serveBuild(worker: ReturnType<typeof createWorker>, label = 'v2') {
  for (const file of BUILD) worker.network.set(at(`./${file}`), () => new FakeResponse(`${file} ${label}`));
}

describe('offline service worker', () => {
  it('pre-caches every build file on install and activates without waiting', async () => {
    const worker = createWorker(BUILD);
    serveBuild(worker);
    await worker.lifecycle('install');

    const cache = worker.stores.get(`${CACHE_PREFIX}v2`)!;
    expect([...cache.entries.keys()].sort()).toEqual(BUILD.map((f) => at(`./${f}`)).sort());
    expect(worker.self.skipWaiting).toHaveBeenCalled();
  });

  it('removes the caches of earlier builds, keeps unrelated caches and claims open pages', async () => {
    const worker = createWorker(BUILD);
    serveBuild(worker);
    await worker.lifecycle('install');
    const current = worker.stores.get(`${CACHE_PREFIX}v2`)!;
    worker.stores.set(`${CACHE_PREFIX}v1`, current);
    worker.stores.set('someone-elses-cache', current);

    await worker.lifecycle('activate');
    expect([...worker.stores.keys()].sort()).toEqual([`${CACHE_PREFIX}v2`, 'someone-elses-cache']);
    expect(worker.self.clients.claim).toHaveBeenCalled();
  });

  it('serves navigations from the network when online and the cached app when offline', async () => {
    const worker = createWorker(BUILD);
    serveBuild(worker);
    await worker.lifecycle('install');

    worker.network.set(at('./'), () => new FakeResponse('index from network'));
    const online = await worker.fetchEvent(request(at('./'), { mode: 'navigate' }));
    expect(await online!.text()).toBe('index from network');

    worker.setOnline(false);
    const again = await worker.fetchEvent(request(at('./'), { mode: 'navigate' }));
    expect(await again!.text()).toBe('index from network');
    const deepLink = await worker.fetchEvent(request(at('./runs/42'), { mode: 'navigate' }));
    expect(await deepLink!.text()).toBe('index.html v2');
  });

  it('serves build files from the cache without touching the network', async () => {
    const worker = createWorker(BUILD);
    serveBuild(worker);
    await worker.lifecycle('install');
    worker.setOnline(false);
    const before = worker.fetched.length;

    const script = await worker.fetchEvent(request(at('./assets/chunk-def.js')));
    expect(await script!.text()).toBe('assets/chunk-def.js v2');
    expect(worker.fetched.length).toBe(before);
  });

  it('caches same-origin files fetched at run time', async () => {
    const worker = createWorker(BUILD);
    serveBuild(worker);
    await worker.lifecycle('install');
    worker.network.set(at('./standalone/page.html'), () => new FakeResponse('page'));
    await worker.fetchEvent(request(at('./standalone/page.html')));

    worker.setOnline(false);
    const cached = await worker.fetchEvent(request(at('./standalone/page.html')));
    expect(await cached!.text()).toBe('page');
  });

  it('does not intercept other origins or non-GET requests', async () => {
    const worker = createWorker(BUILD);
    serveBuild(worker);
    await worker.lifecycle('install');

    expect(await worker.fetchEvent(request('https://elsewhere.example/api'))).toBeUndefined();
    expect(
      await worker.fetchEvent(request(at('./assets/index-abc.js'), { method: 'POST' }))
    ).toBeUndefined();
  });

  it('copies a redirected cached page before it answers an offline navigation', async () => {
    const worker = createWorker(BUILD);
    serveBuild(worker);
    worker.network.set(at('./index.html'), () => new FakeResponse('shell', 200, true));
    await worker.lifecycle('install');
    worker.setOnline(false);

    const response = await worker.fetchEvent(request(at('./offline'), { mode: 'navigate' }));
    expect(response).toBeInstanceOf(Response);
    expect((response as Response).redirected).toBe(false);
    expect(await response!.text()).toBe('shell');
  });

  it('fails to install (and keeps the previous worker) when a build file is missing', async () => {
    const worker = createWorker([...BUILD, 'assets/missing.js']);
    serveBuild(worker);
    await expect(worker.lifecycle('install')).rejects.toThrow('missing.js');
    expect(worker.self.skipWaiting).not.toHaveBeenCalled();
  });
});

describe('service worker build helpers', () => {
  it('pre-caches the build without source maps or the worker itself', () => {
    expect(
      selectPrecacheFiles(['assets/a.js', 'assets/a.js.map', 'sw.js', 'icon.svg', '.DS_Store'])
    ).toEqual(['assets/a.js', 'icon.svg', 'index.html']);
  });

  it('versions builds by content, independently of file order', () => {
    const a = buildVersion(['index.html:x', 'assets/a.js:1']);
    expect(buildVersion(['assets/a.js:1', 'index.html:x'])).toBe(a);
    expect(buildVersion(['index.html:x', 'assets/a.js:2'])).not.toBe(a);
    expect(a).toMatch(/^[0-9a-f]{8}$/);
  });
});
