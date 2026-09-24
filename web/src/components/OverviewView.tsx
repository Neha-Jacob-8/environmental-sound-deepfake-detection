/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { formatEer, formatParams } from '../data/researchData';
import { useResearchData } from '../data/DataContext';
import { AlertCircle, ArrowUpRight, CheckCircle2, ShieldAlert, Cpu } from 'lucide-react';

interface OverviewViewProps {
  onNavigateToModel?: (modelId: string) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({ onNavigateToModel }) => {
  const { MODELS, CONFIDENCE, DATASET_STATS } = useResearchData();
  // Sort models by unseen EER ascending (best to worst)
  const sortedLeaderboard = [...MODELS].sort((a, b) => a.unseenEer - b.unseenEer);

  // Best model is index 0 (logmel_cnn)
  const bestModel = sortedLeaderboard[0];
  const maxEerScale = 0.40; // normalizer for the inline mini-bars

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header & Research Question */}
      <section className="border-b pb-6" style={{ borderColor: 'var(--grid-line)' }}>
        <div className="text-xs font-mono uppercase tracking-wider mb-2" style={{ color: 'var(--text-secondary)' }}>
          EnvSDD &middot; ESDD 2026 protocol
        </div>

        <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-balance leading-snug mb-3">
          Robust Environmental Sound Deepfake Detection Against Unseen Audio Generators
        </h1>

        {/* Primary Research Question in one sentence */}
        <div 
          className="p-4 rounded-md border text-sm md:text-base leading-relaxed"
          style={{ 
            backgroundColor: 'var(--surface-subtle)', 
            borderColor: 'var(--grid-line)' 
          }}
        >
          <span className="font-semibold" style={{ color: 'var(--color-series-1)' }}>Research question: </span>
          <span className="font-medium">
            Can combining complementary acoustic representations improve robustness of
            environmental sound deepfake detection against unseen audio generators?
          </span>
          <div className="mt-3 pt-3 border-t text-sm leading-relaxed" style={{ borderColor: 'var(--grid-line)' }}>
            <span className="font-semibold">Answer: no.</span>{' '}
            Feature fusion did not improve robustness &mdash; it lands between its own two
            branches on seen generators and is worse than the plain waveform CNN on unseen
            ones. The strongest detector remains the simple log-Mel CNN, and it still roughly
            triples its error rate on generators it never trained on.
          </div>
        </div>
      </section>

      {/* Hero Row of 4 Stat Tiles */}
      <section aria-label="Key Headline Metrics">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div 
            className="p-4 rounded-md border"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <div className="text-xs uppercase tracking-wider font-mono mb-1" style={{ color: 'var(--text-secondary)' }}>
              Best Seen EER
            </div>
            <div className="text-2xl md:text-3xl font-mono font-semibold" style={{ color: 'var(--color-seen)' }}>
              {formatEer(bestModel.seenEer)}
            </div>
            <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
              {bestModel.label} (G01–G04)
            </div>
          </div>

          <div 
            className="p-4 rounded-md border"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <div className="text-xs uppercase tracking-wider font-mono mb-1" style={{ color: 'var(--text-secondary)' }}>
              Best Unseen EER
            </div>
            <div className="text-2xl md:text-3xl font-mono font-semibold" style={{ color: 'var(--color-unseen)' }}>
              {formatEer(bestModel.unseenEer)}
            </div>
            <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
              {bestModel.label} (G05–G07)
            </div>
          </div>

          <div 
            className="p-4 rounded-md border"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <div className="text-xs uppercase tracking-wider font-mono mb-1" style={{ color: 'var(--text-secondary)' }}>
              Best Model Gap
            </div>
            <div className="text-2xl md:text-3xl font-mono font-semibold" style={{ color: 'var(--color-series-1)' }}>
              +{formatEer(bestModel.gap)}
            </div>
            <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
              unseen EER − seen EER
            </div>
          </div>

          <div 
            className="p-4 rounded-md border"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <div className="text-xs uppercase tracking-wider font-mono mb-1" style={{ color: 'var(--text-secondary)' }}>
              Total Audio Clips
            </div>
            <div className="text-2xl md:text-3xl font-mono font-semibold">
              {DATASET_STATS.totalClips.toLocaleString()}
            </div>
            <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
              EnvSDD · {DATASET_STATS.sampleRate} · {DATASET_STATS.duration}
            </div>
          </div>
        </div>
      </section>

      {/* Critical Caution Notice */}
      <section aria-label="Caution Notice">
        <div 
          className="flex items-start gap-3 p-4 rounded-md border text-sm leading-relaxed"
          style={{ 
            backgroundColor: 'var(--surface-subtle)', 
            borderColor: 'var(--color-series-4)',
            borderLeftWidth: '4px'
          }}
        >
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" style={{ color: 'var(--color-series-4)' }} />
          <div>
            <span className="font-semibold text-xs uppercase tracking-wider font-mono block mb-0.5" style={{ color: 'var(--color-series-4)' }}>
              Interpretive Caution on Generalisation Gap
            </span>
            <p style={{ color: 'var(--text-primary)' }}>
              A small gap is not automatically good. Fusion's gap looks competitive only because the model is weak everywhere.
              The generalisation gap must be read alongside the seen EER, never alone.
            </p>
          </div>
        </div>
      </section>

      {/* Leaderboard of All Five Models */}
      <section aria-label="Model Leaderboard">
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 mb-3">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">Model Leaderboard</h2>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              Sorted by unseen EER (lower is better). Side-by-side mini bars show seen (orange) vs unseen (aqua) error rates.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-xs" style={{ backgroundColor: 'var(--color-seen)' }} />
              <span>Seen EER</span>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-xs" style={{ backgroundColor: 'var(--color-unseen)' }} />
              <span>Unseen EER</span>
            </span>
          </div>
        </div>

        <div 
          className="rounded-md border overflow-hidden"
          style={{ 
            backgroundColor: 'var(--surface-card)', 
            borderColor: 'var(--grid-line)' 
          }}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm border-collapse min-w-[640px]">
              <thead>
                <tr 
                  className="border-b text-xs font-mono uppercase tracking-wider"
                  style={{ 
                    backgroundColor: 'var(--surface-subtle)', 
                    borderColor: 'var(--grid-line)',
                    color: 'var(--text-secondary)'
                  }}
                >
                  <th scope="col" className="py-2.5 px-3 w-12 text-center">Rank</th>
                  <th scope="col" className="py-2.5 px-3">Model Architecture</th>
                  <th scope="col" className="py-2.5 px-3">Level & Input</th>
                  <th scope="col" className="py-2.5 px-3 text-right">Seen EER</th>
                  <th scope="col" className="py-2.5 px-3 text-right">Unseen EER</th>
                  <th scope="col" className="py-2.5 px-3 text-right">Gap (Δ)</th>
                  <th scope="col" className="py-2.5 px-4 w-52">Visual Comparison</th>
                </tr>
              </thead>
              <tbody className="divide-y" style={{ borderColor: 'var(--grid-line)' }}>
                {sortedLeaderboard.map((model, idx) => {
                  const ci = CONFIDENCE[model.id as keyof typeof CONFIDENCE];
                  const isTop = idx === 0;
                  const seenPct = Math.min(100, (model.seenEer / maxEerScale) * 100);
                  const unseenPct = Math.min(100, (model.unseenEer / maxEerScale) * 100);

                  return (
                    <tr 
                      key={model.id}
                      className="hover:bg-[var(--surface-hover)] transition-colors"
                    >
                      <td className="py-3 px-3 text-center font-mono text-xs">
                        {isTop ? (
                          <span className="font-bold text-xs" style={{ color: 'var(--color-series-1)' }}>#1</span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)' }}>#{idx + 1}</span>
                        )}
                      </td>
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-2">
                          <span 
                            className="w-2.5 h-2.5 rounded-full shrink-0" 
                            style={{ backgroundColor: model.color }}
                            title={`Series color: ${model.color}`}
                          />
                          <span className="font-semibold">{model.label}</span>
                          {isTop && (
                            <span 
                              className="text-[10px] font-mono px-1.5 py-0.5 rounded border"
                              style={{ 
                                color: 'var(--color-series-1)',
                                borderColor: 'var(--color-series-1)'
                              }}
                            >
                              Lowest Error
                            </span>
                          )}
                        </div>
                        <div className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                          {formatParams(model.params)} params
                        </div>
                      </td>
                      <td className="py-3 px-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
                        <div>Level {model.level}</div>
                        <div className="font-mono">{model.input}</div>
                      </td>
                      <td className="py-3 px-3 text-right font-mono font-medium" style={{ color: 'var(--color-seen)' }}>
                        {formatEer(model.seenEer)}
                        {ci && (
                          <div className="text-[10px] font-mono text-muted" style={{ color: 'var(--text-muted)' }}>
                            [{formatEer(ci.seen[0])}, {formatEer(ci.seen[1])}]
                          </div>
                        )}
                      </td>
                      <td className="py-3 px-3 text-right font-mono font-medium" style={{ color: 'var(--color-unseen)' }}>
                        {formatEer(model.unseenEer)}
                        {ci && (
                          <div className="text-[10px] font-mono text-muted" style={{ color: 'var(--text-muted)' }}>
                            [{formatEer(ci.unseen[0])}, {formatEer(ci.unseen[1])}]
                          </div>
                        )}
                      </td>
                      <td className="py-3 px-3 text-right font-mono font-medium">
                        +{formatEer(model.gap)}
                        {ci && (
                          <div className="text-[10px] font-mono text-muted" style={{ color: 'var(--text-muted)' }}>
                            [{formatEer(ci.gap[0])}, {formatEer(ci.gap[1])}]
                          </div>
                        )}
                      </td>
                      <td className="py-3 px-4">
                        <div className="space-y-1 py-1">
                          {/* Seen Bar */}
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono w-7 text-right" style={{ color: 'var(--color-seen)' }}>S</span>
                            <div className="w-full bg-[var(--surface-subtle)] h-2.5 rounded-xs overflow-hidden">
                              <div 
                                className="h-full rounded-xs transition-all duration-300"
                                style={{ 
                                  width: `${seenPct}%`, 
                                  backgroundColor: 'var(--color-seen)' 
                                }}
                              />
                            </div>
                          </div>
                          {/* Unseen Bar */}
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono w-7 text-right" style={{ color: 'var(--color-unseen)' }}>U</span>
                            <div className="w-full bg-[var(--surface-subtle)] h-2.5 rounded-xs overflow-hidden">
                              <div 
                                className="h-full rounded-xs transition-all duration-300"
                                style={{ 
                                  width: `${unseenPct}%`, 
                                  backgroundColor: 'var(--color-unseen)' 
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="p-3 text-xs border-t font-mono flex flex-wrap items-center justify-between gap-2" style={{ borderColor: 'var(--grid-line)', color: 'var(--text-secondary)' }}>
            <span>Values show EER (Equal Error Rate — lower is better). 95% bootstrap confidence intervals shown in brackets [2.5%, 97.5%] for resampled models.</span>
            <span>Scale normalized to 0.40 EER</span>
          </div>
        </div>
      </section>

      {/* The Three Findings Cards verbatim from prompt */}
      <section aria-label="Key Research Findings">
        <h2 className="text-lg font-semibold tracking-tight mb-3">Key Research Findings</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Finding 1 */}
          <article 
            className="p-5 rounded-md border flex flex-col justify-between"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <div>
              <div className="flex items-center gap-2 mb-2 font-mono text-xs uppercase tracking-wider" style={{ color: 'var(--color-series-1)' }}>
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>Finding 1 · Architecture Simplicity</span>
              </div>
              <h3 className="font-semibold text-base mb-2 leading-snug">
                The simplest model wins by a wide margin.
              </h3>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                The log-Mel CNN reaches 0.0242 seen and 0.0833 unseen — roughly 5× better than AASIST and an order of magnitude better than the Level 3 and fusion models. The likely reason is data: 1,200 training source recordings is far too little for graph attention or a frozen self-supervised front-end to pay off.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t text-xs font-mono flex justify-between" style={{ borderColor: 'var(--grid-line)', color: 'var(--text-muted)' }}>
              <span>Log-Mel CNN EER: 0.0242 / 0.0833</span>
              <span>Params: 240,737</span>
            </div>
          </article>

          {/* Finding 2 */}
          <article 
            className="p-5 rounded-md border flex flex-col justify-between"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <div>
              <div className="flex items-center gap-2 mb-2 font-mono text-xs uppercase tracking-wider" style={{ color: 'var(--color-series-5)' }}>
                <Cpu className="w-4 h-4 shrink-0" />
                <span>Finding 2 · Fusion Behaviour</span>
              </div>
              <h3 className="font-semibold text-base mb-2 leading-snug">
                Fusion did not beat its own branches.
              </h3>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                It sits between the waveform CNN and BEATs+AASIST on seen generators and is worse than the waveform CNN on unseen ones. Combining two weak branches did not produce a strong one.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t text-xs font-mono flex justify-between" style={{ borderColor: 'var(--grid-line)', color: 'var(--text-muted)' }}>
              <span>Fusion vs CNN Unseen: 0.2967 vs 0.2700</span>
              <span>Params: 324,227</span>
            </div>
          </article>

          {/* Finding 3 */}
          <article 
            className="p-5 rounded-md border flex flex-col justify-between"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <div>
              <div className="flex items-center gap-2 mb-2 font-mono text-xs uppercase tracking-wider" style={{ color: 'var(--color-series-2)' }}>
                <ShieldAlert className="w-4 h-4 shrink-0" />
                <span>Finding 3 · Conditioning Modes</span>
              </div>
              <h3 className="font-semibold text-base mb-2 leading-snug">
                Three models are at chance on G07.
              </h3>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                The waveform CNN, fusion and BEATs+AASIST all score about 0.51 on G07 — a coin flip. AASIST manages 0.1133 and the log-Mel CNN 0.0833 on the same clips, so the information is there and those three simply fail to use it.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t text-xs font-mono flex justify-between" style={{ borderColor: 'var(--grid-line)', color: 'var(--text-muted)' }}>
              <span>G07 EER: ~0.51 vs 0.0833</span>
              <span>Mode: Audio-to-Audio</span>
            </div>
          </article>
        </div>
      </section>

      {/* Dataset & Benchmark Specifications */}
      <section 
        className="p-5 rounded-md border text-xs"
        style={{ 
          backgroundColor: 'var(--surface-card)', 
          borderColor: 'var(--grid-line)' 
        }}
      >
        <div className="text-xs uppercase tracking-wider font-mono mb-3 font-semibold" style={{ color: 'var(--text-primary)' }}>
          EnvSDD Evaluation Protocol & Dataset Specifications
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 font-mono">
          <div>
            <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Dataset</div>
            <div className="font-semibold mt-0.5">{DATASET_STATS.name}</div>
          </div>
          <div>
            <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Total Clips</div>
            <div className="font-semibold mt-0.5">{DATASET_STATS.totalClips.toLocaleString()}</div>
          </div>
          <div>
            <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Train / Val / Test</div>
            <div className="font-semibold mt-0.5">6,000 / 1,500 / 2,400</div>
          </div>
          <div>
            <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Audio Format</div>
            <div className="font-semibold mt-0.5">{DATASET_STATS.sampleRate} · {DATASET_STATS.duration}</div>
          </div>
          <div>
            <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Generators</div>
            <div className="font-semibold mt-0.5">4 Seen · 3 Unseen</div>
          </div>
          <div>
            <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Bootstrap</div>
            <div className="font-semibold mt-0.5">300 groups · 2,000 reps</div>
          </div>
        </div>
      </section>
    </div>
  );
};
