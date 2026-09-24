/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo } from 'react';
import { ModelInfo, formatEer, formatParams } from '../data/researchData';
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
  Cell,
  LabelList,
} from 'recharts';
import { ArrowUpDown, ArrowUp, ArrowDown, HelpCircle } from 'lucide-react';

type SortField = 'unseenEer' | 'seenEer' | 'gap' | 'params' | 'valEer' | 'label' | 'level';
type SortOrder = 'asc' | 'desc';

interface ModelComparisonViewProps {
  selectedModelId?: string | null;
  onSelectModel?: (id: string | null) => void;
}

export const ModelComparisonView: React.FC<ModelComparisonViewProps> = ({
  selectedModelId: externalSelectedId,
  onSelectModel,
}) => {
  const { MODELS, CONFIDENCE } = useResearchData();
  const [internalSelectedId, setInternalSelectedId] = useState<string | null>(null);
  const selectedModelId = externalSelectedId !== undefined ? externalSelectedId : internalSelectedId;

  const handleSelect = (id: string) => {
    const next = selectedModelId === id ? null : id;
    if (onSelectModel) {
      onSelectModel(next);
    } else {
      setInternalSelectedId(next);
    }
  };

  const [sortField, setSortField] = useState<SortField>('unseenEer');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(prev => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      // Default to ascending for error rates/gap/name, descending for params
      setSortOrder(field === 'params' ? 'desc' : 'asc');
    }
  };

  const sortedData = useMemo(() => {
    return [...MODELS].sort((a, b) => {
      let comparison = 0;
      if (sortField === 'label') {
        comparison = a.label.localeCompare(b.label);
      } else if (sortField === 'level') {
        comparison = a.level.localeCompare(b.level);
      } else {
        comparison = (a[sortField] as number) - (b[sortField] as number);
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });
  }, [sortField, sortOrder]);

  // Transform data for horizontal grouped bar chart
  const chartData = useMemo(() => {
    return sortedData.map(m => ({
      ...m,
      name: m.label,
      seenEerNum: m.seenEer,
      unseenEerNum: m.unseenEer,
    }));
  }, [sortedData]);

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Title & Research Context */}
      <div>
        <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider mb-1" style={{ color: 'var(--text-secondary)' }}>
          <span>Evaluation Metric: EER</span>
          <span>/</span>
          <span>Seen vs Unseen Generators</span>
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Model Comparison & Generalisation Gap
        </h1>
        <p className="text-sm mt-1 text-balance" style={{ color: 'var(--text-secondary)' }}>
          Comparing detector Equal Error Rate (EER) across seen generators (G01–G04) versus unseen generators (G05–G07).
          The generalisation gap (unseen EER − seen EER) quantifies robustness to novel acoustic generators.
        </p>
      </div>

      {/* Control row above chart */}
      <div 
        className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-md border text-xs"
        style={{ 
          backgroundColor: 'var(--surface-card)', 
          borderColor: 'var(--grid-line)' 
        }}
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono uppercase tracking-wider font-semibold" style={{ color: 'var(--text-secondary)' }}>
            Sort Models By:
          </span>
          {[
            { id: 'unseenEer', label: 'Unseen EER' },
            { id: 'seenEer', label: 'Seen EER' },
            { id: 'gap', label: 'Gap (Δ)' },
            { id: 'params', label: 'Parameter Count' },
          ].map(btn => {
            const isActive = sortField === btn.id;
            return (
              <button
                key={btn.id}
                type="button"
                onClick={() => handleSort(btn.id as SortField)}
                className={`px-2.5 py-1 rounded font-mono border transition-colors focus:outline-hidden focus-visible:ring-2 focus-visible:ring-blue-500 cursor-pointer ${
                  isActive
                    ? 'bg-[var(--surface-hover)] font-semibold border-[var(--text-primary)] text-[var(--text-primary)]'
                    : 'border-transparent text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]'
                }`}
                aria-pressed={isActive}
              >
                <span className="flex items-center gap-1">
                  {btn.label}
                  {isActive && (
                    sortOrder === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />
                  )}
                </span>
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-3 font-mono text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {selectedModelId ? (
            <div className="flex items-center gap-2">
              <span>Highlighted: <strong>{MODELS.find(m => m.id === selectedModelId)?.label}</strong></span>
              <button 
                type="button"
                onClick={() => handleSelect(selectedModelId)}
                className="underline hover:text-[var(--text-primary)] cursor-pointer"
              >
                Clear highlight
              </button>
            </div>
          ) : (
            <span>Click any model row or bar to highlight</span>
          )}
        </div>
      </div>

      {/* Chart above table, both full width: the table has eight columns and
          was being clipped when the two sat side by side. */}
      <div className="flex flex-col gap-6">
        {/* Grouped Horizontal Bar Chart */}
        <div 
          className="p-4 rounded-md border"
          style={{ 
            backgroundColor: 'var(--surface-card)', 
            borderColor: 'var(--grid-line)' 
          }}
        >
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="font-semibold text-sm">Grouped EER by Architecture</h2>
              <div className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                Direct comparisons on identical EnvSDD test clips
              </div>
            </div>
            <div className="text-[11px] font-mono text-right" style={{ color: 'var(--text-muted)' }}>
              Unit: EER (lower is better)
            </div>
          </div>

          <div className="h-[360px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={chartData}
                margin={{ top: 10, right: 45, left: 30, bottom: 20 }}
                barGap={3}
                barCategoryGap={16}
              >
                <CartesianGrid 
                  strokeDasharray="2 2" 
                  horizontal={false} 
                  stroke="var(--grid-line)" 
                />
                <XAxis 
                  type="number" 
                  domain={[0, 0.40]} 
                  tickFormatter={(val: number) => val.toFixed(2)}
                  stroke="var(--text-secondary)"
                  tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' }}
                  label={{ 
                    value: 'EER — lower is better', 
                    position: 'insideBottom', 
                    offset: -10,
                    style: { fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' } 
                  }}
                />
                <YAxis 
                  type="category" 
                  dataKey="label" 
                  width={110}
                  stroke="var(--text-secondary)"
                  tick={{ fontSize: 11, fill: 'var(--text-primary)', fontWeight: 500 }}
                />
                <Tooltip
                  cursor={{ fill: 'var(--surface-hover)', opacity: 0.5 }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload as ModelInfo;
                      const ci = CONFIDENCE[data.id as keyof typeof CONFIDENCE];
                      return (
                        <div 
                          className="p-3 rounded border text-xs shadow-xs font-mono"
                          style={{ 
                            backgroundColor: 'var(--surface-card)', 
                            borderColor: 'var(--grid-line)',
                            color: 'var(--text-primary)'
                          }}
                        >
                          <div className="font-semibold text-sm mb-2">{data.label}</div>
                          <div className="space-y-1">
                            <div className="flex justify-between gap-4">
                              <span style={{ color: 'var(--color-seen)' }}>Seen EER (G01–G04):</span>
                              <span className="font-semibold">{formatEer(data.seenEer)}</span>
                            </div>
                            {ci && (
                              <div className="text-[10px] text-right" style={{ color: 'var(--text-muted)' }}>
                                95% CI: [{formatEer(ci.seen[0])}, {formatEer(ci.seen[1])}]
                              </div>
                            )}
                            <div className="flex justify-between gap-4">
                              <span style={{ color: 'var(--color-unseen)' }}>Unseen EER (G05–G07):</span>
                              <span className="font-semibold">{formatEer(data.unseenEer)}</span>
                            </div>
                            {ci && (
                              <div className="text-[10px] text-right" style={{ color: 'var(--text-muted)' }}>
                                95% CI: [{formatEer(ci.unseen[0])}, {formatEer(ci.unseen[1])}]
                              </div>
                            )}
                            <div className="flex justify-between gap-4 pt-1 border-t" style={{ borderColor: 'var(--grid-line)' }}>
                              <span>Generalisation Gap:</span>
                              <span className="font-semibold">+{formatEer(data.gap)}</span>
                            </div>
                            <div className="flex justify-between gap-4">
                              <span>Validation EER:</span>
                              <span>{formatEer(data.valEer)}</span>
                            </div>
                            <div className="flex justify-between gap-4 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                              <span>Parameters:</span>
                              <span>{formatParams(data.params)}</span>
                            </div>
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
                  wrapperStyle={{ paddingBottom: 10, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  formatter={(value) => {
                    return value === 'seenEer' 
                      ? 'Seen generators (G01–G04)' 
                      : 'Unseen generators (G05–G07)';
                  }}
                />
                <Bar isAnimationActive={false} 
                  dataKey="seenEer" 
                  name="seenEer"
                  fill="var(--color-seen)" 
                  radius={[0, 2, 2, 0]}
                  onClick={(entry) => {
                    if (entry && typeof entry.id === 'string') {
                      handleSelect(entry.id);
                    }
                  }}
                  cursor="pointer"
                >
                  <LabelList 
                    dataKey="seenEer" 
                    position="right" 
                    formatter={(val: unknown) => typeof val === 'number' ? formatEer(val) : ''}
                    style={{ fontSize: 10, fill: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' }}
                  />
                  {chartData.map((entry) => {
                    const isSelected = selectedModelId === entry.id;
                    const isAnySelected = selectedModelId !== null;
                    return (
                      <Cell 
                        key={`cell-seen-${entry.id}`} 
                        fill="var(--color-seen)"
                        opacity={isAnySelected ? (isSelected ? 1 : 0.25) : 0.95}
                        stroke={isSelected ? 'var(--text-primary)' : 'none'}
                        strokeWidth={isSelected ? 1.5 : 0}
                      />
                    );
                  })}
                </Bar>
                <Bar isAnimationActive={false} 
                  dataKey="unseenEer" 
                  name="unseenEer"
                  fill="var(--color-unseen)" 
                  radius={[0, 2, 2, 0]}
                  onClick={(entry) => {
                    if (entry && typeof entry.id === 'string') {
                      handleSelect(entry.id);
                    }
                  }}
                  cursor="pointer"
                >
                  <LabelList 
                    dataKey="unseenEer" 
                    position="right" 
                    formatter={(val: unknown) => typeof val === 'number' ? formatEer(val) : ''}
                    style={{ fontSize: 10, fill: 'var(--text-secondary)', fontFamily: 'JetBrains Mono' }}
                  />
                  {chartData.map((entry) => {
                    const isSelected = selectedModelId === entry.id;
                    const isAnySelected = selectedModelId !== null;
                    return (
                      <Cell 
                        key={`cell-unseen-${entry.id}`} 
                        fill="var(--color-unseen)"
                        opacity={isAnySelected ? (isSelected ? 1 : 0.25) : 0.95}
                        stroke={isSelected ? 'var(--text-primary)' : 'none'}
                        strokeWidth={isSelected ? 1.5 : 0}
                      />
                    );
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[11px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>
            * Values labeled on bars are exact EER to 4 decimal places.
          </div>
        </div>

        {/* Interactive Sortable Data Table */}
        <div 
          className="rounded-md border overflow-hidden"
          style={{ 
            backgroundColor: 'var(--surface-card)', 
            borderColor: 'var(--grid-line)' 
          }}
        >
          <div className="p-3 border-b flex items-center justify-between" style={{ borderColor: 'var(--grid-line)' }}>
            <div>
              <h2 className="font-semibold text-sm">Model Architecture Specifications & Results</h2>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                Click headers to sort table. Click a row to highlight across the dashboard.
              </p>
            </div>
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
                  <th scope="col" className="py-2.5 px-3">
                    <button 
                      type="button"
                      onClick={() => handleSort('label')}
                      className="flex items-center gap-1 hover:text-[var(--text-primary)] cursor-pointer"
                    >
                      Model
                      <ArrowUpDown className="w-3 h-3 opacity-50" />
                    </button>
                  </th>
                  <th scope="col" className="py-2.5 px-2">
                    <button 
                      type="button"
                      onClick={() => handleSort('level')}
                      className="flex items-center gap-1 hover:text-[var(--text-primary)] cursor-pointer"
                    >
                      Lvl
                      <ArrowUpDown className="w-3 h-3 opacity-50" />
                    </button>
                  </th>
                  <th scope="col" className="py-2.5 px-2">Input</th>
                  <th scope="col" className="py-2.5 px-2 text-right">
                    <button 
                      type="button"
                      onClick={() => handleSort('params')}
                      className="flex items-center justify-end gap-1 hover:text-[var(--text-primary)] ml-auto cursor-pointer"
                    >
                      Params
                      <ArrowUpDown className="w-3 h-3 opacity-50" />
                    </button>
                  </th>
                  <th scope="col" className="py-2.5 px-2 text-right">
                    <button 
                      type="button"
                      onClick={() => handleSort('valEer')}
                      className="flex items-center justify-end gap-1 hover:text-[var(--text-primary)] ml-auto cursor-pointer"
                    >
                      Val EER
                      <ArrowUpDown className="w-3 h-3 opacity-50" />
                    </button>
                  </th>
                  <th scope="col" className="py-2.5 px-2 text-right">
                    <button 
                      type="button"
                      onClick={() => handleSort('seenEer')}
                      className="flex items-center justify-end gap-1 hover:text-[var(--text-primary)] ml-auto cursor-pointer"
                    >
                      Seen EER
                      <ArrowUpDown className="w-3 h-3 opacity-50" />
                    </button>
                  </th>
                  <th scope="col" className="py-2.5 px-2 text-right">
                    <button 
                      type="button"
                      onClick={() => handleSort('unseenEer')}
                      className="flex items-center justify-end gap-1 hover:text-[var(--text-primary)] ml-auto cursor-pointer"
                    >
                      Unseen EER
                      <ArrowUpDown className="w-3 h-3 opacity-50" />
                    </button>
                  </th>
                  <th scope="col" className="py-2.5 px-3 text-right">
                    <button 
                      type="button"
                      onClick={() => handleSort('gap')}
                      className="flex items-center justify-end gap-1 hover:text-[var(--text-primary)] ml-auto cursor-pointer"
                    >
                      Gap
                      <ArrowUpDown className="w-3 h-3 opacity-50" />
                    </button>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y font-mono" style={{ borderColor: 'var(--grid-line)' }}>
                {sortedData.map((model) => {
                  const isSelected = selectedModelId === model.id;
                  const isTop = model.id === 'logmel_cnn';

                  return (
                    <tr
                      key={model.id}
                      onClick={() => handleSelect(model.id)}
                      className={`cursor-pointer transition-colors ${
                        isSelected 
                          ? 'bg-[var(--surface-hover)] ring-1 ring-inset ring-[var(--text-primary)]' 
                          : 'hover:bg-[var(--surface-hover)]'
                      }`}
                    >
                      <td className="py-2.5 px-3 font-sans">
                        <div className="flex items-center gap-2">
                          <span 
                            className="w-2 h-2 rounded-full shrink-0" 
                            style={{ backgroundColor: model.color }}
                          />
                          <span className={`font-medium ${isTop ? 'font-semibold' : ''}`}>
                            {model.label}
                          </span>
                        </div>
                      </td>
                      <td className="py-2.5 px-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                        {model.level}
                      </td>
                      <td className="py-2.5 px-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                        {model.input}
                      </td>
                      <td className="py-2.5 px-2 text-right text-[11px]">
                        {formatParams(model.params)}
                      </td>
                      <td className="py-2.5 px-2 text-right text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                        {formatEer(model.valEer)}
                      </td>
                      <td className="py-2.5 px-2 text-right font-medium" style={{ color: 'var(--color-seen)' }}>
                        {formatEer(model.seenEer)}
                      </td>
                      <td className="py-2.5 px-2 text-right font-medium" style={{ color: 'var(--color-unseen)' }}>
                        {formatEer(model.unseenEer)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-semibold">
                        +{formatEer(model.gap)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="p-3 border-t text-[11px] font-mono leading-relaxed" style={{ borderColor: 'var(--grid-line)', color: 'var(--text-secondary)' }}>
            <div><strong>Generalisation Gap:</strong> <span className="font-sans">unseen EER − seen EER</span>. A smaller gap indicates higher cross-generator consistency, but must be interpreted alongside absolute error rates.</div>
          </div>
        </div>
      </div>
    </div>
  );
};
