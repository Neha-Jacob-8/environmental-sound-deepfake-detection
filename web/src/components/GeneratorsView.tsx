/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  MODELS,
  GENERATORS,
  PER_GENERATOR,
  formatEer,
} from '../data/researchData';
import { Info, Sparkles, Filter } from 'lucide-react';

export const GeneratorsView: React.FC = () => {
  const [filterMode, setFilterMode] = useState<'all' | 'seen' | 'unseen'>('all');

  const filteredGenerators = GENERATORS.filter(gen => {
    if (filterMode === 'seen') return gen.seen;
    if (filterMode === 'unseen') return !gen.seen;
    return true;
  });

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider mb-1" style={{ color: 'var(--text-secondary)' }}>
          <span>Generative Engines Profile</span>
          <span>/</span>
          <span>G01 to G07 Synthesis Architectures</span>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              Acoustic Generator Profiles & Model Performance
            </h1>
            <p className="text-sm mt-1 text-balance" style={{ color: 'var(--text-secondary)' }}>
              Individual performance profiles for each environmental sound generator. Models trained strictly on G01–G04.
              Generators G05, G06, and G07 were completely withheld until evaluation.
            </p>
          </div>

          {/* Filter Bar */}
          <div 
            className="flex items-center p-1 rounded-md border text-xs font-mono shrink-0 self-start sm:self-auto"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <button
              type="button"
              onClick={() => setFilterMode('all')}
              className={`px-3 py-1 rounded transition-colors cursor-pointer ${
                filterMode === 'all'
                  ? 'bg-[var(--surface-hover)] font-semibold text-[var(--text-primary)] shadow-xs'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              aria-pressed={filterMode === 'all'}
            >
              All (7)
            </button>
            <button
              type="button"
              onClick={() => setFilterMode('seen')}
              className={`px-3 py-1 rounded transition-colors cursor-pointer ${
                filterMode === 'seen'
                  ? 'bg-[var(--surface-hover)] font-semibold text-[var(--text-primary)] shadow-xs'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              aria-pressed={filterMode === 'seen'}
            >
              Seen in Training (4)
            </button>
            <button
              type="button"
              onClick={() => setFilterMode('unseen')}
              className={`px-3 py-1 rounded transition-colors cursor-pointer ${
                filterMode === 'unseen'
                  ? 'bg-[var(--surface-hover)] font-semibold text-[var(--text-primary)] shadow-xs'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              aria-pressed={filterMode === 'unseen'}
            >
              Unseen at Test (3)
            </button>
          </div>
        </div>
      </div>

      {/* Generator Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {filteredGenerators.map(gen => {
          // Collect and sort models by EER on this generator (lowest error first)
          const modelScores = MODELS.map(m => ({
            ...m,
            eer: PER_GENERATOR[m.id][gen.id],
          })).sort((a, b) => a.eer - b.eer);

          const isUnseen = !gen.seen;
          const maxLocalEer = 0.55; // scale for mini-bars

          return (
            <div
              key={gen.id}
              className={`rounded-md border p-5 flex flex-col justify-between transition-all ${
                isUnseen 
                  ? 'border-l-4' 
                  : 'border-l-4'
              }`}
              style={{
                backgroundColor: 'var(--surface-card)',
                borderColor: 'var(--grid-line)',
                borderLeftColor: isUnseen ? 'var(--color-unseen)' : 'var(--color-seen)',
              }}
            >
              <div>
                {/* Header tag and ID */}
                <div className="flex items-center justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-xl font-bold">{gen.id}</span>
                    <span className="text-sm font-semibold font-sans">{gen.name}</span>
                  </div>
                  
                  {/* Seen vs Unseen badge */}
                  <span
                    className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded border"
                    style={{
                      color: isUnseen ? 'var(--color-unseen)' : 'var(--color-seen)',
                      backgroundColor: isUnseen ? 'rgba(27, 175, 122, 0.08)' : 'rgba(235, 104, 52, 0.08)',
                      borderColor: isUnseen ? 'rgba(27, 175, 122, 0.3)' : 'rgba(235, 104, 52, 0.3)',
                    }}
                  >
                    {isUnseen ? 'Unseen at Test' : 'Seen at Training'}
                  </span>
                </div>

                {/* Conditioning mode */}
                <div className="text-xs font-mono mb-3 flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
                  <span>Mode:</span>
                  <span className="font-semibold text-[var(--text-primary)]">{gen.mode}</span>
                  {gen.mode === 'audio-to-audio' && (
                    <span 
                      className="text-[10px] px-1.5 py-0.2 rounded border"
                      style={{ 
                        color: 'var(--color-series-4)', 
                        borderColor: 'var(--color-series-4)' 
                      }}
                    >
                      structure-preserving
                    </span>
                  )}
                </div>

                {/* Specific architectural notes */}
                {gen.notes && (
                  <div 
                    className="p-2.5 rounded text-xs leading-relaxed mb-4 border"
                    style={{ 
                      backgroundColor: 'var(--surface-subtle)', 
                      borderColor: 'var(--grid-line)',
                      color: 'var(--text-secondary)' 
                    }}
                  >
                    <p>{gen.notes}</p>
                  </div>
                )}

                {/* Small sorted bar list */}
                <div className="space-y-2 mt-4">
                  <div className="text-[11px] font-mono uppercase tracking-wider flex justify-between" style={{ color: 'var(--text-muted)' }}>
                    <span>Detectors (Ranked by EER)</span>
                    <span>EER</span>
                  </div>

                  <div className="space-y-1.5">
                    {modelScores.map((score, rankIdx) => {
                      const pct = Math.min(100, (score.eer / maxLocalEer) * 100);
                      const isChance = score.eer >= 0.50;

                      return (
                        <div key={score.id} className="text-xs font-mono">
                          <div className="flex items-center justify-between mb-0.5">
                            <div className="flex items-center gap-1.5">
                              <span className="text-[10px] text-muted w-3">{rankIdx + 1}.</span>
                              <span 
                                className="w-2 h-2 rounded-full shrink-0" 
                                style={{ backgroundColor: score.color }} 
                              />
                              <span className="font-sans font-medium text-[11px] truncate max-w-[130px]">
                                {score.label}
                              </span>
                            </div>
                            <div className="flex items-center gap-1.5">
                              {isChance && (
                                <span className="text-[9px] text-red-500 font-mono">chance</span>
                              )}
                              <span className={`font-semibold ${rankIdx === 0 ? 'text-[var(--color-series-1)]' : ''}`}>
                                {formatEer(score.eer)}
                              </span>
                            </div>
                          </div>

                          {/* Mini Bar */}
                          <div className="w-full bg-[var(--surface-subtle)] h-1.5 rounded-xs overflow-hidden">
                            <div
                              className="h-full rounded-xs transition-all duration-300"
                              style={{
                                width: `${pct}%`,
                                backgroundColor: isChance ? 'var(--color-series-2)' : score.color,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Bottom Card Summary */}
              <div 
                className="mt-4 pt-3 border-t text-[11px] font-mono flex items-center justify-between"
                style={{ borderColor: 'var(--grid-line)', color: 'var(--text-muted)' }}
              >
                <span>Best: <strong>{modelScores[0].label}</strong></span>
                <span>EER: {formatEer(modelScores[0].eer)}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Structural Distinction Callout Banner */}
      <section 
        className="p-4 rounded-md border text-xs leading-relaxed"
        style={{ 
          backgroundColor: 'var(--surface-card)', 
          borderColor: 'var(--grid-line)' 
        }}
      >
        <div className="flex items-start gap-2.5">
          <Info className="w-4 h-4 shrink-0 mt-0.5" style={{ color: 'var(--color-series-1)' }} />
          <div>
            <span className="font-semibold block font-mono mb-1">
              Methodological Note on Conditioning Modes & G07 Unseen Status
            </span>
            <p style={{ color: 'var(--text-secondary)' }}>
              G07 is the same architecture as G02 (AudioLDM 2) but operated in a different conditioning mode (audio-to-audio vs text-to-audio).
              Hence, G07 represents an <em>unseen conditioning mode</em> rather than an unseen network architecture.
              Audio-to-audio generators (G04, G07) preserve the source recording's underlying acoustic structure, making synthetic artifacts significantly harder for waveform detectors to isolate.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
