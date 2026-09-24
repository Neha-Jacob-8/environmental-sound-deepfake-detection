/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import {
  LayoutDashboard,
  BarChart3,
  Grid3X3,
  LineChart as LineChartIcon,
  AudioWaveform,
  Sun,
  Moon,
  Monitor,
  Volume2,
} from 'lucide-react';
import { ThemeMode } from '../hooks/useTheme';

export type ViewId = 'overview' | 'comparison' | 'breakdown' | 'curves' | 'generators';

interface NavigationProps {
  currentView: ViewId;
  onSelectView: (view: ViewId) => void;
  themeMode: ThemeMode;
  onSetThemeMode: (mode: ThemeMode) => void;
}

interface NavItem {
  id: ViewId;
  label: string;
  sublabel: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'overview',   label: 'Overview',           sublabel: 'Headline findings & leaderboard', icon: LayoutDashboard },
  { id: 'comparison', label: 'Model Comparison',   sublabel: 'Seen vs unseen gap analysis',     icon: BarChart3 },
  { id: 'breakdown',  label: 'Generator Breakdown',sublabel: '5×7 matrix & per-gen EER',        icon: Grid3X3 },
  { id: 'curves',     label: 'Training Curves',    sublabel: 'Validation EER & loss traces',    icon: LineChartIcon },
  { id: 'generators', label: 'Generators',         sublabel: 'G01–G07 engine profiles',         icon: AudioWaveform },
];

export const Navigation: React.FC<NavigationProps> = ({
  currentView,
  onSelectView,
  themeMode,
  onSetThemeMode,
}) => {
  return (
    <>
      {/* DESKTOP SIDEBAR (>= 900px) */}
      <aside 
        className="hidden min-[900px]:flex flex-col justify-between w-64 shrink-0 h-screen sticky top-0 border-r p-4 overflow-y-auto"
        style={{ 
          backgroundColor: 'var(--surface-card)', 
          borderColor: 'var(--grid-line)' 
        }}
        aria-label="Main Navigation Sidebar"
      >
        <div>
          {/* Research Brand Header */}
          <div className="mb-6 pb-4 border-b" style={{ borderColor: 'var(--grid-line)' }}>
            <div className="flex items-center gap-2 mb-1">
              <span 
                className="w-2.5 h-2.5 rounded-full" 
                style={{ backgroundColor: 'var(--color-series-1)' }} 
              />
              <span className="font-mono text-xs uppercase tracking-wider font-semibold" style={{ color: 'var(--text-secondary)' }}>
                ESDD 2026 protocol
              </span>
            </div>
            <h2 className="font-semibold text-base tracking-tight leading-snug">
              Audio Deepfake Detection
            </h2>
            <div className="text-[11px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>
              Generalisation across seen vs unseen generators
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = currentView === item.id;

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelectView(item.id)}
                  className={`w-full text-left p-2.5 rounded-md flex items-start gap-3 transition-colors cursor-pointer focus:outline-hidden focus-visible:ring-2 focus-visible:ring-blue-500 ${
                    isActive
                      ? 'bg-[var(--surface-hover)] border border-[var(--surface-border)] font-medium text-[var(--text-primary)]'
                      : 'border border-transparent text-[var(--text-secondary)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]'
                  }`}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <Icon className={`w-4 h-4 mt-0.5 shrink-0 ${isActive ? 'text-[var(--color-series-1)]' : ''}`} />
                  <div>
                    <div className="text-xs font-semibold leading-tight">{item.label}</div>
                    <div className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                      {item.sublabel}
                    </div>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Footer / Theme & Protocol Meta */}
        <div className="pt-4 border-t space-y-3" style={{ borderColor: 'var(--grid-line)' }}>
          {/* Theme Selector: System, Light, Dark */}
          <div>
            <div className="text-[11px] font-mono uppercase tracking-wider mb-1.5" style={{ color: 'var(--text-muted)' }}>
              Color Theme
            </div>
            <div 
              className="grid grid-cols-3 p-0.5 rounded border text-[11px] font-mono"
              style={{ 
                backgroundColor: 'var(--surface-subtle)', 
                borderColor: 'var(--grid-line)' 
              }}
            >
              <button
                type="button"
                onClick={() => onSetThemeMode('system')}
                className={`py-1 px-1.5 flex items-center justify-center gap-1 rounded transition-colors cursor-pointer ${
                  themeMode === 'system'
                    ? 'bg-[var(--surface-card)] text-[var(--text-primary)] font-semibold shadow-xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
                title="Follow system preference"
                aria-label="Follow system color preference"
              >
                <Monitor className="w-3 h-3" />
                <span>Auto</span>
              </button>
              <button
                type="button"
                onClick={() => onSetThemeMode('light')}
                className={`py-1 px-1.5 flex items-center justify-center gap-1 rounded transition-colors cursor-pointer ${
                  themeMode === 'light'
                    ? 'bg-[var(--surface-card)] text-[var(--text-primary)] font-semibold shadow-xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
                title="Light mode"
                aria-label="Force light mode"
              >
                <Sun className="w-3 h-3" />
                <span>Light</span>
              </button>
              <button
                type="button"
                onClick={() => onSetThemeMode('dark')}
                className={`py-1 px-1.5 flex items-center justify-center gap-1 rounded transition-colors cursor-pointer ${
                  themeMode === 'dark'
                    ? 'bg-[var(--surface-card)] text-[var(--text-primary)] font-semibold shadow-xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
                title="Dark mode"
                aria-label="Force dark mode"
              >
                <Moon className="w-3 h-3" />
                <span>Dark</span>
              </button>
            </div>
          </div>

          <div className="text-[10px] font-mono leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            EnvSDD &middot; CC BY 4.0
            <br />
            9,900 clips · 4.000 s mono
          </div>
        </div>
      </aside>

      {/* MOBILE / TABLET TOP BAR (< 900px) */}
      <header 
        className="min-[900px]:hidden sticky top-0 z-30 border-b p-3 shadow-xs"
        style={{ 
          backgroundColor: 'var(--surface-card)', 
          borderColor: 'var(--grid-line)' 
        }}
        aria-label="Mobile Navigation Bar"
      >
        <div className="flex items-center justify-between gap-3 mb-2.5">
          <div className="flex items-center gap-2">
            <span 
              className="w-2 h-2 rounded-full" 
              style={{ backgroundColor: 'var(--color-series-1)' }} 
            />
            <span className="font-semibold text-sm tracking-tight truncate">
              EnvSDD Audio Deepfake Detection
            </span>
          </div>

          {/* Compact theme toggle */}
          <div 
            className="flex items-center p-0.5 rounded border text-[11px] font-mono shrink-0"
            style={{ 
              backgroundColor: 'var(--surface-subtle)', 
              borderColor: 'var(--grid-line)' 
            }}
          >
            <button
              type="button"
              onClick={() => onSetThemeMode(themeMode === 'dark' ? 'light' : 'dark')}
              className="px-2 py-0.5 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer flex items-center gap-1"
              aria-label="Toggle color mode"
            >
              {themeMode === 'dark' ? <Sun className="w-3 h-3" /> : <Moon className="w-3 h-3" />}
              <span className="capitalize">{themeMode === 'system' ? 'Auto' : themeMode}</span>
            </button>
          </div>
        </div>

        {/* Horizontal scrollable tab buttons */}
        <nav className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs font-mono">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = currentView === item.id;

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectView(item.id)}
                className={`px-2.5 py-1 rounded whitespace-nowrap flex items-center gap-1.5 border transition-colors cursor-pointer focus:outline-hidden focus-visible:ring-2 focus-visible:ring-blue-500 ${
                  isActive
                    ? 'bg-[var(--surface-hover)] border-[var(--text-primary)] font-semibold text-[var(--text-primary)]'
                    : 'border-transparent text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]'
                }`}
                aria-current={isActive ? 'page' : undefined}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-[var(--color-series-1)]' : ''}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </header>
    </>
  );
};
