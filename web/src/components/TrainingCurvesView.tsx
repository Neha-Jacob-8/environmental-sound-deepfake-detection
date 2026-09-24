/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo } from 'react';
import {
  MODELS,
  CURVES,
  formatEer,
} from '../data/researchData';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { TrendingDown, Activity, Table as TableIcon } from 'lucide-react';

export const TrainingCurvesView: React.FC = () => {
  const [metric, setMetric] = useState<'eer' | 'loss'>('eer');
  const [showTable, setShowTable] = useState(false);

  // Model visibility state
  const [visibleModels, setVisibleModels] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    MODELS.forEach(m => {
      initial[m.id] = true;
    });
    return initial;
  });

  const toggleModel = (id: string) => {
    setVisibleModels(prev => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const selectAll = () => {
    const next: Record<string, boolean> = {};
    MODELS.forEach(m => { next[m.id] = true; });
    setVisibleModels(next);
  };

  // Build epochs 1..31
  const epochData = useMemo(() => {
    const data = [];
    for (let epoch = 1; epoch <= 31; epoch++) {
      const row: Record<string, number | undefined> = { epoch };
      MODELS.forEach(m => {
        const arr = metric === 'eer' ? CURVES[m.id].valEer : CURVES[m.id].trainLoss;
        const idx = epoch - 1;
        if (idx < arr.length) {
          row[m.id] = arr[idx];
        } else {
          row[m.id] = undefined;
        }
      });
      data.push(row);
    }
    return data;
  }, [metric]);

  const activeMetricLabel = metric === 'eer' ? 'Validation EER — lower is better' : 'Training Loss';

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider mb-1" style={{ color: 'var(--text-secondary)' }}>
          <span>Epoch-by-Epoch Dynamics</span>
          <span>/</span>
          <span>Convergence & Generalisation</span>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              Training & Validation Curves
            </h1>
            <p className="text-sm mt-1 text-balance" style={{ color: 'var(--text-secondary)' }}>
              Validation EER and training loss trajectories across epochs. Notice architectures ran different training regimes
              (31, 30, 30, 15, 15 epochs) — series stop naturally without artificial interpolation or padding.
            </p>
          </div>

          {/* Metric Selector Toggle (Separate panels/toggles, NEVER twin axes) */}
          <div 
            className="flex items-center p-1 rounded-md border text-xs font-mono shrink-0 self-start sm:self-auto"
            style={{ 
              backgroundColor: 'var(--surface-card)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <button
              type="button"
              onClick={() => setMetric('eer')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded transition-colors cursor-pointer ${
                metric === 'eer'
                  ? 'bg-[var(--surface-hover)] font-semibold text-[var(--text-primary)] shadow-xs'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              aria-pressed={metric === 'eer'}
            >
              <TrendingDown className="w-3.5 h-3.5" />
              <span>Validation EER</span>
            </button>
            <button
              type="button"
              onClick={() => setMetric('loss')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded transition-colors cursor-pointer ${
                metric === 'loss'
                  ? 'bg-[var(--surface-hover)] font-semibold text-[var(--text-primary)] shadow-xs'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              aria-pressed={metric === 'loss'}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>Training Loss</span>
            </button>
          </div>
        </div>
      </div>

      {/* Model Series Checkboxes (Fixed Colors Preserved!) */}
      <div 
        className="p-3.5 rounded-md border flex flex-wrap items-center justify-between gap-3 text-xs"
        style={{ 
          backgroundColor: 'var(--surface-card)', 
          borderColor: 'var(--grid-line)' 
        }}
      >
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-mono uppercase tracking-wider font-semibold" style={{ color: 'var(--text-secondary)' }}>
            Active Series:
          </span>
          {MODELS.map(m => {
            const isChecked = !!visibleModels[m.id];
            const maxEpochs = CURVES[m.id].valEer.length;
            return (
              <label
                key={m.id}
                className="flex items-center gap-2 cursor-pointer select-none font-medium px-2 py-1 rounded hover:bg-[var(--surface-hover)] transition-colors"
              >
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => toggleModel(m.id)}
                  className="rounded border-[var(--surface-border)] text-blue-600 focus:ring-blue-500 w-3.5 h-3.5 cursor-pointer"
                />
                <span 
                  className="w-2.5 h-2.5 rounded-full shrink-0" 
                  style={{ backgroundColor: m.color }} 
                />
                <span style={{ color: isChecked ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                  {m.label}
                </span>
                <span className="text-[10px] font-mono text-muted" style={{ color: 'var(--text-muted)' }}>
                  ({maxEpochs} ep)
                </span>
              </label>
            );
          })}
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <button
            type="button"
            onClick={selectAll}
            className="px-2 py-0.5 rounded border text-[11px] hover:bg-[var(--surface-hover)] cursor-pointer"
            style={{ borderColor: 'var(--grid-line)' }}
          >
            Show All
          </button>
          <button
            type="button"
            onClick={() => setShowTable(prev => !prev)}
            className="px-2 py-0.5 rounded border text-[11px] flex items-center gap-1 hover:bg-[var(--surface-hover)] cursor-pointer"
            style={{ borderColor: 'var(--grid-line)' }}
          >
            <TableIcon className="w-3 h-3" />
            <span>{showTable ? 'Hide Table' : 'Show Table'}</span>
          </button>
        </div>
      </div>

      {/* Main Line Plot */}
      <div 
        className="p-4 rounded-md border"
        style={{ 
          backgroundColor: 'var(--surface-card)', 
          borderColor: 'var(--grid-line)' 
        }}
      >
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="font-semibold text-sm">
              {metric === 'eer' ? 'Validation Equal Error Rate by Epoch' : 'Cross-Entropy Training Loss by Epoch'}
            </h2>
            <div className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
              {metric === 'eer'
                ? 'Rapid divergence: Log-Mel CNN settles near 0.0033 val EER while others plateau above 0.08'
                : 'Steady loss decay without overfitting symptoms'}
            </div>
          </div>
          <div className="text-[11px] font-mono text-right" style={{ color: 'var(--text-muted)' }}>
            Hover for crosshair readout
          </div>
        </div>

        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={epochData}
              margin={{ top: 10, right: 30, left: 10, bottom: 25 }}
            >
              <CartesianGrid strokeDasharray="2 2" stroke="var(--grid-line)" />
              <XAxis 
                dataKey="epoch" 
                type="number"
                domain={[1, 31]}
                tickCount={16}
                stroke="var(--text-secondary)"
                tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' }}
                label={{ 
                  value: 'epoch', 
                  position: 'insideBottom', 
                  offset: -12,
                  style: { fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' } 
                }}
              />
              <YAxis 
                domain={metric === 'eer' ? [0, 0.35] : [0, 0.65]}
                tickFormatter={(val: number) => val.toFixed(2)}
                stroke="var(--text-secondary)"
                tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' }}
                label={{ 
                  value: activeMetricLabel, 
                  angle: -90, 
                  position: 'insideLeft', 
                  offset: 10,
                  style: { fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' } 
                }}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div 
                        className="p-3 rounded border text-xs shadow-xs font-mono"
                        style={{ 
                          backgroundColor: 'var(--surface-card)', 
                          borderColor: 'var(--grid-line)',
                          color: 'var(--text-primary)'
                        }}
                      >
                        <div className="font-semibold text-sm mb-1.5 border-b pb-1" style={{ borderColor: 'var(--grid-line)' }}>
                          Epoch {label}
                        </div>
                        <div className="space-y-1">
                          {payload.map(p => {
                            const m = MODELS.find(mod => mod.id === p.dataKey);
                            if (p.value === undefined) return null;
                            return (
                              <div key={p.dataKey as string} className="flex justify-between items-center gap-4">
                                <span className="flex items-center gap-1.5" style={{ color: p.color }}>
                                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
                                  <span>{m?.label}:</span>
                                </span>
                                <span className="font-semibold">
                                  {formatEer(p.value as number)}
                                </span>
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
                iconType="plainline"
                wrapperStyle={{ paddingBottom: 10, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                formatter={(val) => {
                  const m = MODELS.find(mod => mod.id === val);
                  return m ? m.label : val;
                }}
              />
              {MODELS.map(m => {
                if (!visibleModels[m.id]) return null;
                return (
                  <Line
                    key={m.id}
                    type="monotone"
                    dataKey={m.id}
                    name={m.id}
                    stroke={m.color}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, stroke: m.color, strokeWidth: 1, fill: 'var(--surface-card)' }}
                    connectNulls={false}
                    isAnimationActive={true}
                    animationDuration={200}
                  />
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Epoch Data Table view */}
      {showTable && (
        <div 
          className="rounded-md border overflow-hidden animate-fadeIn"
          style={{ 
            backgroundColor: 'var(--surface-card)', 
            borderColor: 'var(--grid-line)' 
          }}
        >
          <div className="p-3 border-b flex items-center justify-between" style={{ borderColor: 'var(--grid-line)' }}>
            <h3 className="font-semibold text-xs font-mono uppercase">
              Epoch Tabular Readout ({metric === 'eer' ? 'Val EER' : 'Train Loss'})
            </h3>
            <span className="text-[11px] font-mono" style={{ color: 'var(--text-secondary)' }}>
              Exact 4-decimal precision across all epochs
            </span>
          </div>
          <div className="overflow-x-auto max-h-72">
            <table className="w-full text-left text-xs border-collapse font-mono">
              <thead className="sticky top-0 shadow-xs">
                <tr 
                  className="border-b uppercase text-[11px]"
                  style={{ 
                    backgroundColor: 'var(--surface-subtle)', 
                    borderColor: 'var(--grid-line)',
                    color: 'var(--text-secondary)'
                  }}
                >
                  <th className="py-2 px-3">Epoch</th>
                  {MODELS.map(m => (
                    <th key={m.id} className="py-2 px-3 text-right">
                      {m.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y" style={{ borderColor: 'var(--grid-line)' }}>
                {epochData.map(row => (
                  <tr key={row.epoch} className="hover:bg-[var(--surface-hover)]">
                    <td className="py-1.5 px-3 font-semibold">{row.epoch}</td>
                    {MODELS.map(m => {
                      const val = row[m.id];
                      return (
                        <td key={m.id} className="py-1.5 px-3 text-right">
                          {val !== undefined ? formatEer(val) : <span style={{ color: 'var(--text-muted)' }}>—</span>}
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
