/**
 * Fetches the results bundle once and hands it to every view.
 *
 * Components previously imported hard-coded constants. They now read the same
 * names off this context, so their bodies are unchanged but the values arrive
 * from the API at runtime.
 */

import React, {createContext, useContext, useEffect, useState} from 'react';
import {Bundle, fetchBundle} from '../api/client';

interface ResearchData {
  MODELS: Bundle['models'];
  GENERATORS: Bundle['generators'];
  PER_GENERATOR: Bundle['perGenerator'];
  CURVES: Bundle['curves'];
  CONFIDENCE: Bundle['confidence'];
  DATASET_STATS: NonNullable<Bundle['dataset']>;
  SUMMARY: Bundle['summary'];
}

const Ctx = createContext<ResearchData | null>(null);

export const useResearchData = (): ResearchData => {
  const v = useContext(Ctx);
  // Views only ever render inside a resolved provider, so a null here is a
  // wiring mistake, not a loading state - fail loudly rather than render
  // empty charts that look like real results.
  if (!v) throw new Error('useResearchData used outside <ResearchDataProvider>');
  return v;
};

export const ResearchDataProvider: React.FC<{children: React.ReactNode}> = ({children}) => {
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const ac = new AbortController();
    setError(null);
    fetchBundle(ac.signal)
      .then(setBundle)
      .catch((e) => {
        if (e.name !== 'AbortError') setError(e.message);
      });
    return () => ac.abort();
  }, [attempt]);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div
          className="max-w-lg rounded-md border p-6 space-y-3"
          style={{borderColor: 'var(--color-status-warning, #eda100)'}}>
          <h2 className="font-semibold">Cannot reach the results API</h2>
          <p className="text-sm" style={{color: 'var(--text-secondary)'}}>{error}</p>
          <p className="text-sm" style={{color: 'var(--text-secondary)'}}>
            Start it from the repository root:
          </p>
          <pre className="text-xs font-mono p-3 rounded overflow-x-auto"
               style={{backgroundColor: 'var(--surface-subtle)'}}>
python3 -m src.api.main</pre>
          <button
            onClick={() => setAttempt((a) => a + 1)}
            className="text-sm px-3 py-1.5 rounded border"
            style={{borderColor: 'var(--grid-line)'}}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  // The dashboard describes a dataset throughout; rendering it with the
  // counts blanked out would be worse than saying plainly what is missing.
  if (bundle && !bundle.dataset) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="max-w-lg rounded-md border p-6 space-y-3"
             style={{borderColor: 'var(--grid-line)'}}>
          <h2 className="font-semibold">Dataset manifest not found</h2>
          <p className="text-sm" style={{color: 'var(--text-secondary)'}}>
            The API is running but <code>data/metadata/manifest.csv</code> is
            missing, so clip counts cannot be reported. Build it with:
          </p>
          <pre className="text-xs font-mono p-3 rounded overflow-x-auto"
               style={{backgroundColor: 'var(--surface-subtle)'}}>
python3 -m src.preprocessing.verify_subset</pre>
        </div>
      </div>
    );
  }

  if (!bundle) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-sm font-mono" style={{color: 'var(--text-secondary)'}}>
          loading results…
        </p>
      </div>
    );
  }

  return (
    <Ctx.Provider
      value={{
        MODELS: bundle.models,
        GENERATORS: bundle.generators,
        PER_GENERATOR: bundle.perGenerator,
        CURVES: bundle.curves,
        CONFIDENCE: bundle.confidence,
        DATASET_STATS: bundle.dataset!,
        SUMMARY: bundle.summary,
      }}>
      {children}
    </Ctx.Provider>
  );
};
