import React, { useState, useEffect, useRef } from 'react';
import { 
  Columns, Database, Sparkles, Sliders, Moon, Sun, 
  HelpCircle, RefreshCw, Layers, ShieldCheck
} from 'lucide-react';
import CommandPanel from './components/CommandPanel';
import ResultGrid from './components/ResultGrid';
import VideoModal from './components/VideoModal';
import HelpModal from './components/HelpModal';

export default function App() {
  const [query, setQuery] = useState('kho vàng thỏi và nhân viên kiểm đếm');
  const [dataset, setDataset] = useState('newsseek');
  const [strategy, setStrategy] = useState('dual');
  const [topK, setTopK] = useState(12);
  const [viewMode, setViewMode] = useState('dual');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [inspectedItem, setInspectedItem] = useState(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [theme, setTheme] = useState('dark');

  const [statusData, setStatusData] = useState(null);
  const [datasetsInfo, setDatasetsInfo] = useState([]);
  const [benchmarkData, setBenchmarkData] = useState(null);

  // Switch dataset and synchronize view mode
  const handleDatasetChange = (newDataset) => {
    setDataset(newDataset);
    if (newDataset === 'coco') {
      setViewMode('benchmark');
    } else if (newDataset === 'newsseek') {
      if (viewMode === 'benchmark') {
        setViewMode('dual');
      }
    }
  };

  // Keyboard shortcut listener ('/' to focus)
  useEffect(() => {
    const handleGlobalKeyDown = (e) => {
      if (e.key === '/' && document.activeElement?.tagName !== 'TEXTAREA' && document.activeElement?.tagName !== 'INPUT') {
        e.preventDefault();
        const el = document.querySelector('textarea');
        if (el) el.focus();
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, []);

  // Sync HTML dark class
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  // Load Status and Datasets on startup
  useEffect(() => {
    fetch('/api/status')
      .then((r) => r.json())
      .then((d) => setStatusData(d))
      .catch((e) => console.error('Status fetch error:', e));

    fetch('/api/datasets')
      .then((r) => r.json())
      .then((d) => setDatasetsInfo(d.datasets || []))
      .catch((e) => console.error('Datasets fetch error:', e));
  }, []);

  // Fetch benchmark data when dataset changes or when benchmark view is selected
  useEffect(() => {
    fetch(`/api/benchmark?dataset=${dataset}`)
      .then((r) => r.json())
      .then((d) => setBenchmarkData(d))
      .catch((e) => console.error('Benchmark fetch error:', e));
  }, [dataset]);

  // Execute Text Search
  const handleSearch = async (customQuery, customDataset) => {
    const q = (typeof customQuery === 'string' ? customQuery : query).trim();
    const d = typeof customDataset === 'string' ? customDataset : dataset;
    if (!q) return;
    if (typeof customQuery === 'string') setQuery(customQuery);
    if (typeof customDataset === 'string') setDataset(customDataset);

    setLoading(true);
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          dataset: d,
          k: topK,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Search request failed');
      }
      const data = await res.json();
      setResults(data);
      if (viewMode === 'benchmark') {
        setViewMode('dual');
      }
    } catch (e) {
      alert(`Search error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Execute Reverse Image Search
  const handleImageSearch = async (file) => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('dataset', dataset);
    formData.append('k', topK);

    try {
      const res = await fetch('/api/search-image', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Image search failed');
      }
      const data = await res.json();
      setResults(data);
      if (viewMode === 'benchmark') {
        setViewMode('dual');
      }
    } catch (e) {
      alert(`Image search error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Preset queries for active dataset
  const activeDatasetObj = datasetsInfo.find((d) => d.id === dataset);
  const currentPresets = activeDatasetObj?.presets || [];

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#211a14] text-[#ece1cd]">
      {/* Top Navigation Bar */}
      <header className="flex h-12 flex-shrink-0 items-center justify-between border-b-2 border-[#5c4d3a] bg-[#2a221b] px-4 select-none">
        {/* App Title & Stamp Logo */}
        <div className="flex items-center gap-3">
          <div className="flex h-8 items-center gap-2 rounded border-2 border-[#5c4d3a] bg-[#211a14] px-2.5 shadow-[2px_2px_0_0_#000]">
            <span className="flex h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="font-mono text-xs font-bold tracking-wider text-[#ece1cd] uppercase">
              RETRIEVAL WORKSTATION
            </span>
            <span className="rounded bg-[#ea580c] px-1.5 py-0.2 font-mono text-[9px] font-bold text-white uppercase">
              v2.4
            </span>
          </div>

          <span className="hidden lg:inline text-xs font-mono text-[#a69580]">
            CLIP (512D) vs SigLIP (768D) Comparative Engine
          </span>
        </div>

        {/* System & Hardware Status Badges */}
        <div className="flex items-center gap-2.5 font-mono text-xs">
          {/* Accelerator Status */}
          <div className="hidden sm:flex items-center gap-1.5 rounded border border-[#5c4d3a] bg-[#211a14] px-2.5 py-1">
            <span className="text-[#a69580]">GPU:</span>
            <span className="font-bold text-emerald-400">
              {statusData?.device_name || 'NVIDIA CUDA'}
            </span>
          </div>

          {/* Dataset Status */}
          <div className="hidden md:flex items-center gap-1.5 rounded border border-[#5c4d3a] bg-[#211a14] px-2.5 py-1">
            <span className="text-[#a69580]">Stores:</span>
            <span className="text-[#f59e0b] font-bold">
              COCO (5K)
            </span>
            <span className="text-[#5c4d3a]">|</span>
            <span className={statusData?.datasets?.newsseek?.ready ? "text-emerald-400 font-bold" : "text-[#a69580]"}>
              NewsSeek ({statusData?.datasets?.newsseek?.ready ? "Ready" : "Colab A100"})
            </span>
          </div>

          {/* Theme Toggle */}
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="flex h-8 w-8 items-center justify-center rounded border border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:text-[#ece1cd] hover:border-[#f59e0b] transition-colors"
            title="Toggle theme"
          >
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
          </button>

          {/* Help Button */}
          <button
            onClick={() => setHelpOpen(true)}
            className="flex h-8 w-8 items-center justify-center rounded border border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:text-[#ece1cd] hover:border-[#f59e0b] transition-colors"
            title="Keyboard Shortcuts & Guide"
          >
            <HelpCircle size={14} />
          </button>
        </div>
      </header>

      {/* Main Workspace Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Command Panel (Sidebar) */}
        <CommandPanel
          query={query}
          setQuery={setQuery}
          dataset={dataset}
          setDataset={handleDatasetChange}
          strategy={strategy}
          setStrategy={setStrategy}
          topK={topK}
          setTopK={setTopK}
          loading={loading}
          onSearch={handleSearch}
          onImageSearch={handleImageSearch}
          presets={currentPresets}
          datasetsInfo={datasetsInfo}
          collapsed={collapsed}
          setCollapsed={setCollapsed}
          onOpenHelp={() => setHelpOpen(true)}
          onOpenBenchmark={() => setViewMode('benchmark')}
        />

        {/* Right Results & Benchmark Workstation */}
        <ResultGrid
          viewMode={viewMode}
          setViewMode={setViewMode}
          results={results}
          loading={loading}
          benchmarkData={benchmarkData}
          onInspect={(item) => setInspectedItem(item)}
          onOpenBenchmark={() => setViewMode('benchmark')}
        />
      </div>

      {/* Frame Inspector Modal */}
      {inspectedItem && (
        <VideoModal
          item={inspectedItem}
          onClose={() => setInspectedItem(null)}
          onReverseSearch={(imgUrl) => {
            // Can trigger search or copy
          }}
        />
      )}

      {/* System Guide & Shortcuts Modal */}
      <HelpModal isOpen={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  );
}
