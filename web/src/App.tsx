/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useState } from 'react';
import { Navigation, ViewId } from './components/Navigation';
import { OverviewView } from './components/OverviewView';
import { ModelComparisonView } from './components/ModelComparisonView';
import { GeneratorBreakdownView } from './components/GeneratorBreakdownView';
import { TrainingCurvesView } from './components/TrainingCurvesView';
import { GeneratorsView } from './components/GeneratorsView';
import { ResearchDataProvider } from './data/DataContext';
import { useTheme } from './hooks/useTheme';

const VIEWS: ViewId[] = ['overview', 'comparison', 'breakdown', 'curves', 'generators'];

const viewFromHash = (): ViewId => {
  const h = window.location.hash.replace(/^#/, '') as ViewId;
  return VIEWS.includes(h) ? h : 'overview';
};

export default function App() {
  // Views live in the URL hash so a particular chart can be linked to, and so
  // reloading does not bounce you back to the overview.
  const [currentView, setCurrentViewState] = useState<ViewId>(viewFromHash);

  const setCurrentView = (v: ViewId) => {
    setCurrentViewState(v);
    if (viewFromHash() !== v) window.location.hash = v;
  };

  // Back/forward and hand-edited URLs.
  useEffect(() => {
    const onHash = () => setCurrentViewState(viewFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const { mode, setMode, resolvedTheme } = useTheme();

  const isDark = resolvedTheme === 'dark';

  const handleNavigateToModel = (modelId: string) => {
    setSelectedModelId(modelId);
    setCurrentView('comparison');
  };

  return (
    <ResearchDataProvider>
      <div className="min-h-screen flex flex-col min-[900px]:flex-row" style={{ backgroundColor: 'var(--surface)', color: 'var(--text-primary)' }}>
        {/* Responsive Navigation: Left Sidebar on >=900px, Top bar on <900px */}
        <Navigation
          currentView={currentView}
          onSelectView={setCurrentView}
          themeMode={mode}
          onSetThemeMode={setMode}
        />

        {/* Main Content Area */}
        {/* min-w-0: a flex item defaults to min-width:auto and will not shrink
            below its content, so a wide table or chart pushes the whole page
            wider than the viewport instead of scrolling inside its own card. */}
        <main className="flex-1 min-w-0 w-full max-w-6xl mx-auto p-4 sm:p-6 lg:p-8 overflow-y-auto">
          <div className="transition-opacity duration-150 ease-out">
            {currentView === 'overview' && (
              <OverviewView onNavigateToModel={handleNavigateToModel} />
            )}

            {currentView === 'comparison' && (
              <ModelComparisonView
                selectedModelId={selectedModelId}
                onSelectModel={setSelectedModelId}
              />
            )}

            {currentView === 'breakdown' && (
              <GeneratorBreakdownView isDark={isDark} />
            )}

            {currentView === 'curves' && (
              <TrainingCurvesView />
            )}

            {currentView === 'generators' && (
              <GeneratorsView />
            )}
          </div>

          {/* Quiet research footer */}
          <footer 
            className="mt-16 pt-6 border-t flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono text-muted"
            style={{ borderColor: 'var(--grid-line)', color: 'var(--text-muted)' }}
          >
            <div>
              EnvSDD Research Dashboard · Environmental Sound Deepfake Detection
            </div>
            <div>
              Equal Error Rate (EER) Metrics · 4.000s Audio @ 16 kHz Mono
            </div>
          </footer>
        </main>
      </div>
    </ResearchDataProvider>
  );
}
