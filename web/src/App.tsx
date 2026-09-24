/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Navigation, ViewId } from './components/Navigation';
import { OverviewView } from './components/OverviewView';
import { ModelComparisonView } from './components/ModelComparisonView';
import { GeneratorBreakdownView } from './components/GeneratorBreakdownView';
import { TrainingCurvesView } from './components/TrainingCurvesView';
import { GeneratorsView } from './components/GeneratorsView';
import { ResearchDataProvider } from './data/DataContext';
import { useTheme } from './hooks/useTheme';

export default function App() {
  const [currentView, setCurrentView] = useState<ViewId>('overview');
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
        <main className="flex-1 w-full max-w-6xl mx-auto p-4 sm:p-6 lg:p-8 overflow-y-auto">
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
