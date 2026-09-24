/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo } from 'react';
import { formatEer } from '../data/researchData';
import { useResearchData } from '../data/DataContext';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { LayoutGrid, BarChart2, Table as TableIcon, Info } from 'lucide-react';

interface GeneratorBreakdownViewProps {
  isDark?: boolean;
}

export const GeneratorBreakdownView: React.FC<GeneratorBreakdownViewProps> = ({ isDark }) => {
  const { MODELS, GENERATORS, PER_GENERATOR } = useResearchData();
  const [viewMode, setViewMode] = useState<'matrix' | 'bars' | 'table'>('matrix');
  const [hoveredCell, setHoveredCell] = useState<{
    modelId: string;
    modelLabel: string;
    genId: string;
    genName: string;
    genMode: string;
    genSeen: boolean;
    eer: number;
  } | null>(null);

  // Split generators into seen (G01-G04) and unseen (G05-G07)
  const seenGenerators = GENERATORS.filter(g => g.seen);
  const unseenGenerators = GENERATORS.filter(g => !g.seen);

  // Colour-scale bounds come from the data on screen. Pinning them to today's
  // numbers would silently mis-colour the matrix, and mislabel the legend, the
  // first time anything is retrained.
  const allEers = Object.values(PER_GENERATOR).flatMap((row) => Object.values(row));
  const minEer = allEers.length ? Math.min(...allEers) : 0;
  const maxEer = allEers.length ? Math.max(...allEers) : 1;
  // EER is a rate where 0.5 is a coin flip, whatever the data says.
  const CHANCE = 0.5;

  // Single-hue sequential ramp based on #2a78d6 (light = low = good, dark = high = worse)
  const getCellStyles = (eer: number) => {
    // normalized 0 to 1
    const t = Math.max(0, Math.min(1, (eer - minEer) / (maxEer - minEer)));

    if (isDark) {
      // Dark mode: background from dark subtle blue to vibrant deep blue
      // t=0: rgba(42, 120, 214, 0.08)
      // t=1: rgba(42, 120, 214, 0.75)
      const alpha = 0.08 + t * 0.72;
      const bg = `rgba(42, 120, 214, ${alpha.toFixed(3)})`;
      const textColor = t > 0.4 ? '#ffffff' : '#e2e8f0';
      return { backgroundColor: bg, color: textColor };
    } else {
      // Light mode:
      // t=0: very light tint rgba(42, 120, 214, 0.06)
      // t=1: deep solid blue rgba(42, 120, 214, 0.90)
      const alpha = 0.06 + t * 0.82;
      const bg = `rgba(42, 120, 214, ${alpha.toFixed(3)})`;
      const textColor = t > 0.52 ? '#ffffff' : '#0b0b0b';
      return { backgroundColor: bg, color: textColor };
    }
  };

  // Prepare grouped bar chart data (group by generator)
  const groupedBarData = useMemo(() => {
    return GENERATORS.map(gen => {
      const row: Record<string, unknown> = {
        id: gen.id,
        name: `${gen.id} (${gen.name})`,
        genName: gen.name,
        seen: gen.seen,
        mode: gen.mode,
      };
      MODELS.forEach(m => {
        row[m.id] = PER_GENERATOR[m.id][gen.id];
      });
      return row;
    });
  }, []);

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider mb-1" style={{ color: 'var(--text-secondary)' }}>
          <span>Granular Generator Matrix</span>
          <span>/</span>
          <span>7 Audio Synthesis Engines</span>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              EER per Generator Breakdown
            </h1>
            <p className="text-sm mt-1 text-balance" style={{ color: 'var(--text-secondary)' }}>
              A 5×7 evaluation matrix mapping every model against all seen (G01–G04) and unseen (G05–G07) generators.
              Light cells indicate lower error rates (better detection); darker cells indicate chance-level failures.
            </p>
          </div>

          {/* Mode Switcher */}
          <div 
            className="flex items-center p-1 rounded-md border text-xs font-mono shrink-0 self-start sm:self-auto"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <button
              type="button"
              onClick={() => setViewMode('matrix')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded transition-colors cursor-pointer ${
                viewMode === 'matrix'
                  ? 'bg-[var(--surface-hover)] font-semibold text-[var(--text-primary)] shadow-xs'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              aria-pressed={viewMode === 'matrix'}
            >
              <LayoutGrid className="w-3.5 h-3.5" />
              <span>5×7 Matrix</span>
            </button>
            <button
              type="button"
              onClick={() => setViewMode('bars')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded transition-colors cursor-pointer ${
                viewMode === 'bars'
                  ? 'bg-[var(--surface-hover)] font-semibold text-[var(--text-primary)] shadow-xs'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              aria-pressed={viewMode === 'bars'}
            >
              <BarChart2 className="w-3.5 h-3.5" />
              <span>Grouped Bars</span>
            </button>
            <button
              type="button"
              onClick={() => setViewMode('table')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded transition-colors cursor-pointer ${
                viewMode === 'table'
                  ? 'bg-[var(--surface-hover)] font-semibold text-[var(--text-primary)] shadow-xs'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              aria-pressed={viewMode === 'table'}
            >
              <TableIcon className="w-3.5 h-3.5" />
              <span>Table</span>
            </button>
          </div>
        </div>
      </div>

      {/* MATRIX VIEW */}
      {viewMode === 'matrix' && (
        <div className="space-y-4">
          {/* Matrix Container (with horizontal internal scroll so page does not scroll) */}
          <div 
            className="rounded-md border overflow-hidden"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-center text-xs border-collapse min-w-[760px]">
                {/* Super-headers for Seen vs Unseen */}
                <thead>
                  <tr 
                    className="border-b text-xs font-mono uppercase tracking-wider"
                    style={{ 
                      backgroundColor: 'var(--surface-subtle)', 
                      borderColor: 'var(--grid-line)' 
                    }}
                  >
                    <th 
                      scope="col" 
                      className="py-2.5 px-3 text-left w-48 font-semibold border-r"
                      style={{ borderColor: 'var(--grid-line)', color: 'var(--text-secondary)' }}
                    >
                      Detector Model
                    </th>
                    <th 
                      colSpan={4} 
                      scope="colgroup"
                      className="py-2 px-2 text-center border-r font-semibold"
                      style={{ 
                        borderColor: 'var(--color-seen)',
                        borderRightWidth: '3px',
                        color: 'var(--color-seen)',
                        backgroundColor: 'rgba(235, 104, 52, 0.05)'
                      }}
                    >
                      Seen at Training (G01 – G04)
                    </th>
                    <th 
                      colSpan={3} 
                      scope="colgroup"
                      className="py-2 px-2 text-center font-semibold"
                      style={{ 
                        color: 'var(--color-unseen)',
                        backgroundColor: 'rgba(27, 175, 122, 0.05)'
                      }}
                    >
                      Unseen at Test Time (G05 – G07)
                    </th>
                  </tr>

                  {/* Sub-headers for individual generators */}
                  <tr 
                    className="border-b font-mono text-[11px]"
                    style={{ 
                      backgroundColor: 'var(--surface-card)', 
                      borderColor: 'var(--grid-line)' 
                    }}
                  >
                    <th className="py-2 px-3 text-left border-r text-xs font-sans text-muted" style={{ borderColor: 'var(--grid-line)', color: 'var(--text-muted)' }}>
                      Input representation
                    </th>
                    {seenGenerators.map(gen => (
                      <th 
                        key={gen.id} 
                        className="py-2 px-2 border-r"
                        style={{ borderColor: 'var(--grid-line)' }}
                      >
                        <div className="font-semibold text-xs font-mono">{gen.id}</div>
                        <div className="text-[10px] font-sans truncate max-w-[85px] mx-auto" style={{ color: 'var(--text-secondary)' }}>
                          {gen.name}
                        </div>
                        <div className="text-[9px] font-mono text-muted uppercase tracking-tighter" style={{ color: 'var(--text-muted)' }}>
                          {gen.mode === 'audio-to-audio' ? 'aud→aud' : 'txt→aud'}
                        </div>
                      </th>
                    ))}
                    {unseenGenerators.map((gen, idx) => (
                      <th 
                        key={gen.id} 
                        className={`py-2 px-2 ${idx < unseenGenerators.length - 1 ? 'border-r' : ''}`}
                        style={{ borderColor: 'var(--grid-line)' }}
                      >
                        <div className="font-semibold text-xs font-mono flex items-center justify-center gap-1">
                          <span>{gen.id}</span>
                          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: 'var(--color-unseen)' }} />
                        </div>
                        <div className="text-[10px] font-sans truncate max-w-[85px] mx-auto" style={{ color: 'var(--text-secondary)' }}>
                          {gen.name}
                        </div>
                        <div className="text-[9px] font-mono text-muted uppercase tracking-tighter" style={{ color: 'var(--text-muted)' }}>
                          {gen.mode === 'audio-to-audio' ? 'aud→aud' : 'txt→aud'}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>

                {/* Body Rows */}
                <tbody className="divide-y font-mono" style={{ borderColor: 'var(--grid-line)' }}>
                  {MODELS.map(model => {
                    const rowEers = PER_GENERATOR[model.id];

                    return (
                      <tr key={model.id} className="hover:opacity-95 transition-opacity">
                        <th 
                          scope="row" 
                          className="py-3 px-3 text-left font-sans border-r"
                          style={{ 
                            borderColor: 'var(--grid-line)',
                            backgroundColor: 'var(--surface-subtle)' 
                          }}
                        >
                          <div className="flex items-center gap-2">
                            <span 
                              className="w-2.5 h-2.5 rounded-full shrink-0" 
                              style={{ backgroundColor: model.color }}
                            />
                            <span className="font-semibold text-xs">{model.label}</span>
                          </div>
                          <div className="text-[10px] font-mono mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                            Level {model.level} · {model.input}
                          </div>
                        </th>

                        {/* Seen Generator Cells */}
                        {seenGenerators.map((gen, gIdx) => {
                          const eer = rowEers[gen.id];
                          const styles = getCellStyles(eer);
                          const isLastSeen = gIdx === seenGenerators.length - 1;

                          return (
                            <td
                              key={gen.id}
                              onMouseEnter={() => setHoveredCell({
                                modelId: model.id,
                                modelLabel: model.label,
                                genId: gen.id,
                                genName: gen.name,
                                genMode: gen.mode,
                                genSeen: gen.seen,
                                eer,
                              })}
                              onMouseLeave={() => setHoveredCell(null)}
                              className={`py-3 px-2 transition-transform select-none cursor-pointer ${
                                isLastSeen ? 'border-r-3' : 'border-r'
                              }`}
                              style={{
                                ...styles,
                                borderColor: isLastSeen ? 'var(--color-seen)' : 'var(--grid-line)',
                              }}
                            >
                              <div className="font-semibold text-xs">{formatEer(eer)}</div>
                            </td>
                          );
                        })}

                        {/* Unseen Generator Cells */}
                        {unseenGenerators.map((gen, gIdx) => {
                          const eer = rowEers[gen.id];
                          const styles = getCellStyles(eer);
                          const isG07 = gen.id === 'G07';
                          const isChance = eer > 0.50;

                          return (
                            <td
                              key={gen.id}
                              onMouseEnter={() => setHoveredCell({
                                modelId: model.id,
                                modelLabel: model.label,
                                genId: gen.id,
                                genName: gen.name,
                                genMode: gen.mode,
                                genSeen: gen.seen,
                                eer,
                              })}
                              onMouseLeave={() => setHoveredCell(null)}
                              className={`py-3 px-2 transition-transform select-none cursor-pointer ${
                                gIdx < unseenGenerators.length - 1 ? 'border-r' : ''
                              }`}
                              style={{
                                ...styles,
                                borderColor: 'var(--grid-line)',
                              }}
                            >
                              <div className="flex flex-col items-center">
                                <span className="font-semibold text-xs">{formatEer(eer)}</span>
                                {isG07 && isChance && (
                                  <span 
                                    className="text-[9px] font-mono tracking-tighter uppercase px-1 py-0.2 rounded mt-0.5"
                                    style={{ 
                                      backgroundColor: 'rgba(0, 0, 0, 0.4)', 
                                      color: '#ffffff' 
                                    }}
                                  >
                                    chance (~{formatEer(CHANCE)})
                                  </span>
                                )}
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Matrix Legend & Sequential Ramp Guide */}
            <div 
              className="p-3 border-t text-xs font-mono flex flex-wrap items-center justify-between gap-4"
              style={{ 
                borderColor: 'var(--grid-line)', 
                backgroundColor: 'var(--surface-subtle)',
                color: 'var(--text-secondary)'
              }}
            >
              <div className="flex items-center gap-3">
                <span>EER Sequential Ramp:</span>
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px]">0.0100 (good)</span>
                  <div 
                    className="w-32 h-3.5 rounded-xs border"
                    style={{ 
                      borderColor: 'var(--grid-line)',
                      background: isDark 
                        ? 'linear-gradient(to right, rgba(42, 120, 214, 0.08), rgba(42, 120, 214, 0.75))'
                        : 'linear-gradient(to right, rgba(42, 120, 214, 0.06), rgba(42, 120, 214, 0.90))'
                    }}
                  />
                  <span className="text-[11px]">{formatEer(maxEer)}{maxEer >= CHANCE ? " (chance)" : ""}</span>
                </div>
              </div>

              <div className="flex items-center gap-4 text-[11px]">
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs" style={{ backgroundColor: 'var(--color-seen)' }} />
                  <span>Seen Block (G01–G04)</span>
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs" style={{ backgroundColor: 'var(--color-unseen)' }} />
                  <span>Unseen Block (G05–G07)</span>
                </span>
              </div>
            </div>
          </div>

          {/* Hover readout card */}
          {hoveredCell ? (
            <div 
              className="p-3 rounded-md border flex items-center justify-between text-xs font-mono animate-fadeIn"
              style={{ 
                backgroundColor: 'var(--surface-card)', 
                borderColor: 'var(--color-series-1)' 
              }}
            >
              <div className="flex items-center gap-3">
                <span className="font-semibold text-sm">{hoveredCell.modelLabel}</span>
                <span>on</span>
                <span className="font-bold px-1.5 py-0.5 rounded border" style={{ borderColor: 'var(--grid-line)' }}>
                  {hoveredCell.genId}: {hoveredCell.genName}
                </span>
                <span className="text-secondary">({hoveredCell.genMode})</span>
                <span 
                  className="px-1.5 py-0.5 rounded font-semibold"
                  style={{ 
                    color: hoveredCell.genSeen ? 'var(--color-seen)' : 'var(--color-unseen)',
                    backgroundColor: hoveredCell.genSeen ? 'rgba(235, 104, 52, 0.1)' : 'rgba(27, 175, 122, 0.1)'
                  }}
                >
                  {hoveredCell.genSeen ? 'Seen in Training' : 'Unseen at Test Time'}
                </span>
              </div>
              <div className="text-right">
                <span className="text-secondary mr-2">EER:</span>
                <span className="text-base font-bold" style={{ color: 'var(--color-series-1)' }}>
                  {formatEer(hoveredCell.eer)}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-xs font-mono flex items-center gap-2 p-2 text-muted" style={{ color: 'var(--text-muted)' }}>
              <Info className="w-3.5 h-3.5 shrink-0" />
              <span>Hover over any matrix cell to inspect model, generator details, conditioning mode, and exact EER.</span>
            </div>
          )}
        </div>
      )}

      {/* GROUPED BARS VIEW */}
      {viewMode === 'bars' && (
        <div 
          className="p-4 rounded-md border"
          style={{ 
            backgroundColor: 'var(--surface-card)', 
            borderColor: 'var(--grid-line)' 
          }}
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
            <div>
              <h2 className="font-semibold text-sm">Grouped EER Across Generators</h2>
              <p className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                Comparing all 5 detectors on each generator individually (G01–G04 seen, G05–G07 unseen).
              </p>
            </div>
            <div className="text-[11px] font-mono text-right" style={{ color: 'var(--text-muted)' }}>
              Unit: EER (lower is better)
            </div>
          </div>

          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={groupedBarData}
                margin={{ top: 15, right: 20, left: 10, bottom: 25 }}
                barGap={2}
                barCategoryGap={16}
              >
                <CartesianGrid strokeDasharray="2 2" stroke="var(--grid-line)" vertical={false} />
                <XAxis 
                  dataKey="name" 
                  stroke="var(--text-secondary)"
                  tick={{ fontSize: 11, fill: 'var(--text-primary)', fontFamily: 'JetBrains Mono' }}
                />
                <YAxis 
                  domain={[0, 0.55]} 
                  tickFormatter={(val: number) => val.toFixed(2)}
                  stroke="var(--text-secondary)"
                  tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' }}
                  label={{ 
                    value: 'EER — lower is better', 
                    angle: -90, 
                    position: 'insideLeft', 
                    offset: 10,
                    style: { fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' } 
                  }}
                />
                <Tooltip
                  cursor={{ fill: 'var(--surface-hover)', opacity: 0.5 }}
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const genId = payload[0].payload.id as string;
                      const genObj = GENERATORS.find(g => g.id === genId);
                      return (
                        <div 
                          className="p-3 rounded border text-xs shadow-xs font-mono"
                          style={{ 
                            backgroundColor: 'var(--surface-card)', 
                            borderColor: 'var(--grid-line)',
                            color: 'var(--text-primary)'
                          }}
                        >
                          <div className="font-semibold text-sm">{label}</div>
                          <div className="text-[11px] mb-2" style={{ color: genObj?.seen ? 'var(--color-seen)' : 'var(--color-unseen)' }}>
                            {genObj?.seen ? 'Seen Generator' : 'Unseen Generator'} · {genObj?.mode}
                          </div>
                          <div className="space-y-1">
                            {payload.map(p => {
                              const m = MODELS.find(mod => mod.id === p.dataKey);
                              return (
                                <div key={p.dataKey as string} className="flex justify-between gap-4">
                                  <span style={{ color: p.color }}>{m?.label}:</span>
                                  <span className="font-semibold">{formatEer(p.value as number)}</span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend 
                  verticalAlign="top" 
                  align="right"
                  iconType="square"
                  iconSize={10}
                  wrapperStyle={{ paddingBottom: 15, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  formatter={(val) => {
                    const m = MODELS.find(mod => mod.id === val);
                    return m ? m.label : val;
                  }}
                />
                {MODELS.map(m => (
                  <Bar 
                    key={m.id} 
                    dataKey={m.id} 
                    name={m.id} 
                    fill={m.color} 
                    radius={[2, 2, 0, 0]} 
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* RAW DATA TABLE VIEW */}
      {viewMode === 'table' && (
        <div 
          className="rounded-md border overflow-hidden"
          style={{ 
            backgroundColor: 'var(--surface-card)', 
            borderColor: 'var(--grid-line)' 
          }}
        >
          <div className="p-3 border-b flex items-center justify-between" style={{ borderColor: 'var(--grid-line)' }}>
            <h2 className="font-semibold text-sm">Full Generator EER Data Table</h2>
            <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>All 35 discrete measurements (4 decimals)</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr 
                  className="border-b font-mono uppercase tracking-wider text-[11px]"
                  style={{ 
                    backgroundColor: 'var(--surface-subtle)', 
                    borderColor: 'var(--grid-line)',
                    color: 'var(--text-secondary)'
                  }}
                >
                  <th className="py-2.5 px-3">Generator</th>
                  <th className="py-2.5 px-3">Synthesis Model</th>
                  <th className="py-2.5 px-2">Mode</th>
                  <th className="py-2.5 px-2">Protocol Status</th>
                  {MODELS.map(m => (
                    <th key={m.id} className="py-2.5 px-2 text-right">
                      {m.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y font-mono" style={{ borderColor: 'var(--grid-line)' }}>
                {GENERATORS.map(gen => (
                  <tr key={gen.id} className="hover:bg-[var(--surface-hover)] transition-colors">
                    <td className="py-2.5 px-3 font-semibold">{gen.id}</td>
                    <td className="py-2.5 px-3 font-sans font-medium">{gen.name}</td>
                    <td className="py-2.5 px-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>{gen.mode}</td>
                    <td className="py-2.5 px-2">
                      <span 
                        className="text-[10px] px-1.5 py-0.5 rounded font-semibold"
                        style={{ 
                          color: gen.seen ? 'var(--color-seen)' : 'var(--color-unseen)',
                          backgroundColor: gen.seen ? 'rgba(235, 104, 52, 0.1)' : 'rgba(27, 175, 122, 0.1)'
                        }}
                      >
                        {gen.seen ? 'Seen (Train)' : 'Unseen (Test)'}
                      </span>
                    </td>
                    {MODELS.map(m => {
                      const val = PER_GENERATOR[m.id][gen.id];
                      return (
                        <td 
                          key={m.id} 
                          className="py-2.5 px-2 text-right"
                          style={{ 
                            color: val > 0.5 ? 'var(--color-series-2)' : 'var(--text-primary)',
                            fontWeight: val < 0.05 ? 'bold' : 'normal'
                          }}
                        >
                          {formatEer(val)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
