import React from 'react';
import { 
  Grid, Columns, Film, BarChart3, Clock, Zap, CheckCircle2, 
  HelpCircle, Sparkles, AlertCircle
} from 'lucide-react';
import ResultCard from './ResultCard';
import VideoGroupGrid from './VideoGroupGrid';
import BenchmarkView from './BenchmarkView';

export default function ResultGrid({
  viewMode,
  setViewMode,
  results,
  loading,
  benchmarkData,
  onInspect,
  onOpenBenchmark,
}) {
  const clipResults = results?.clip_results || [];
  const siglipResults = results?.siglip_results || [];
  const fusedResults = results?.fused_results || [];
  const queryBenchmark = results?.benchmark || {};

  const activeResults = fusedResults.length > 0 ? fusedResults : (siglipResults.length > 0 ? siglipResults : clipResults);
  const activeModelName = fusedResults.length > 0 ? 'fusion' : (siglipResults.length > 0 ? 'siglip' : 'clip');

  const clipMetrics = queryBenchmark.clip || {};
  const siglipMetrics = queryBenchmark.siglip || {};
  const agreement = queryBenchmark.agreement || {};

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#211a14] text-[#ece1cd]">
      {/* Telemetry & View Toolbar */}
      <div className="flex flex-wrap items-center justify-between border-b-2 border-[#5c4d3a] bg-[#2a221b] px-4 py-2.5 gap-3 select-none">
        {/* Left: View Mode Tabs */}
        <div className="flex items-center gap-1.5 font-mono text-xs">
          <button
            onClick={() => setViewMode('dual')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded border transition-all ${
              viewMode === 'dual'
                ? 'border-[#f59e0b] bg-[#3a3026] text-[#ece1cd] font-bold shadow-[2px_2px_0_0_#000]'
                : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
            }`}
          >
            <Columns size={13} />
            <span>Dual Side-by-Side</span>
          </button>

          <button
            onClick={() => setViewMode('flat')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded border transition-all ${
              viewMode === 'flat'
                ? 'border-[#f59e0b] bg-[#3a3026] text-[#ece1cd] font-bold shadow-[2px_2px_0_0_#000]'
                : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
            }`}
          >
            <Grid size={13} />
            <span>Dense Grid</span>
          </button>

          <button
            onClick={() => setViewMode('video')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded border transition-all ${
              viewMode === 'video'
                ? 'border-[#f59e0b] bg-[#3a3026] text-[#ece1cd] font-bold shadow-[2px_2px_0_0_#000]'
                : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
            }`}
          >
            <Film size={13} />
            <span>Video Grouped</span>
          </button>

          <button
            onClick={() => setViewMode('benchmark')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded border transition-all ${
              viewMode === 'benchmark'
                ? 'border-[#ea580c] bg-[#ea580c] text-white font-bold shadow-[2px_2px_0_0_#000]'
                : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#ea580c]'
            }`}
          >
            <BarChart3 size={13} />
            <span>Benchmark Report</span>
          </button>
        </div>

        {/* Right: Live Telemetry Gauges */}
        {results && (
          <div className="flex items-center gap-3 font-mono text-[11px]">
            {/* Latency */}
            <div className="flex items-center gap-1 px-2 py-1 rounded border border-[#5c4d3a] bg-[#211a14]">
              <Clock size={12} className="text-[#f59e0b]" />
              <span className="text-[#a69580]">Latency:</span>
              <span className="text-blue-400 font-bold">{clipMetrics.total_latency_ms || 0}ms</span>
              <span className="text-[#5c4d3a]">/</span>
              <span className="text-emerald-400 font-bold">{siglipMetrics.total_latency_ms || 0}ms</span>
            </div>

            {/* Margin */}
            <div className="flex items-center gap-1 px-2 py-1 rounded border border-[#5c4d3a] bg-[#211a14]">
              <Zap size={12} className="text-[#ea580c]" />
              <span className="text-[#a69580]">Sep. Margin:</span>
              <span className="text-emerald-400 font-bold">
                +{((siglipMetrics.separation_margin || 0) * 100).toFixed(1)}%
              </span>
            </div>

            {/* Rank Agreement */}
            {agreement.top5_overlap !== undefined && (
              <div className="flex items-center gap-1 px-2 py-1 rounded border border-[#5c4d3a] bg-[#211a14]">
                <CheckCircle2 size={12} className="text-teal-400" />
                <span className="text-[#a69580]">Top-5 Agreement:</span>
                <span className="font-bold text-[#ece1cd]">{agreement.top5_overlap}/5</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Main Results Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 scrollbar-thin">
        {loading ? (
          <div className="flex h-full flex-col items-center justify-center gap-3">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-[#ea580c] border-t-transparent"></div>
            <p className="font-mono text-xs text-[#a69580]">Encoding cross-modal query on GPU & searching FAISS stores...</p>
          </div>
        ) : viewMode === 'benchmark' ? (
          <BenchmarkView benchmarkData={benchmarkData} />
        ) : !results ? (
          /* Empty Initial State */
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center p-8 max-w-lg mx-auto">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border-2 border-[#5c4d3a] bg-[#2a221b] text-[#f59e0b] shadow-[3px_3px_0_0_#000]">
              <Sparkles size={28} />
            </div>
            <div>
              <h2 className="text-base font-bold font-mono text-[#ece1cd]">
                Multimodal Video Retrieval Workstation
              </h2>
              <p className="mt-1 text-xs text-[#a69580] leading-relaxed">
                Hệ thống truy xuất ngữ nghĩa trực tiếp đa phương thức kết hợp song song hai kiến trúc vector: 
                <span className="font-bold text-blue-400"> CLIP ViT-B/32 (512D)</span> và 
                <span className="font-bold text-emerald-400"> SigLIP ViT-B/16-224 (768D)</span>.
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2 text-xs font-mono text-[#a69580]">
              <span className="px-2 py-1 rounded border border-[#3d3228] bg-[#2a221b]">Press / to focus query</span>
              <span className="px-2 py-1 rounded border border-[#3d3228] bg-[#2a221b]">Ctrl+Enter to search</span>
              <span className="px-2 py-1 rounded border border-[#3d3228] bg-[#2a221b]">Click frame for deep inspection</span>
            </div>
          </div>
        ) : viewMode === 'dual' ? (
          /* Dual Comparative View */
          <div className="flex flex-col gap-4">
            {/* Live Model Comparison Banner */}
            <div className="flex flex-wrap items-center justify-between rounded border-2 border-[#5c4d3a] bg-[#2a221b] p-3 shadow-[2px_2px_0_0_#000] gap-3">
              {/* Query Profile Taxonomy */}
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs font-bold text-[#ece1cd]">Query Profile:</span>
                <span className="rounded bg-[#ea580c]/20 border border-[#ea580c] px-2 py-0.5 font-mono text-[10px] font-bold text-[#ea580c] uppercase">
                  {results.query_profile?.language === 'vi' ? 'Tiếng Việt (VI)' : (results.query_profile?.language === 'en' ? 'English (EN)' : 'Multilingual')}
                </span>
                <span className="rounded bg-[#3a3026] border border-[#5c4d3a] px-2 py-0.5 font-mono text-[10px] text-[#ece1cd] uppercase">
                  {results.query_profile?.granularity || 'Phrase'}
                </span>
                {results.query_profile?.has_relational_cue && (
                  <span className="rounded bg-teal-950/80 border border-teal-600 px-2 py-0.5 font-mono text-[10px] font-bold text-teal-300">
                    Relational / Spatial Cue
                  </span>
                )}
              </div>

              {/* Comparative Metrics Summary */}
              <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
                {/* Score comparison */}
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-[#5c4d3a] bg-[#211a14]">
                  <span className="text-[#a69580]">Top-1 Score:</span>
                  <span className="text-blue-400 font-bold">{((clipMetrics.top1_score || 0) * 100).toFixed(1)}%</span>
                  <span className="text-[#a69580]">vs</span>
                  <span className="text-emerald-400 font-bold">{((siglipMetrics.top1_score || 0) * 100).toFixed(1)}%</span>
                </div>

                {/* Separation Margin */}
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-[#5c4d3a] bg-[#211a14]">
                  <span className="text-[#a69580]">Margin (Δ):</span>
                  <span className="text-blue-400 font-bold">{((clipMetrics.separation_margin || 0) * 100).toFixed(2)}%</span>
                  <span className="text-[#a69580]">/</span>
                  <span className="text-emerald-400 font-bold">{((siglipMetrics.separation_margin || 0) * 100).toFixed(2)}%</span>
                </div>

                {/* Rank 1 Agreement */}
                <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded border ${agreement.rank_1_match ? 'border-teal-600 bg-teal-950/80 text-teal-300' : 'border-amber-600 bg-amber-950/80 text-amber-300'}`}>
                  <span>Rank-1 Agreement:</span>
                  <span className="font-bold">{agreement.rank_1_match ? 'Identical Match' : 'Divergent Perspectives'}</span>
                </div>
              </div>
            </div>

            {/* Side-by-Side Dual Columns */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Left Column: CLIP */}
              <div className="flex flex-col gap-3 rounded border-2 border-blue-900/60 bg-[#251d16] p-3 shadow-[2px_2px_0_0_#000]">
                <div className="flex items-center justify-between border-b border-blue-900/60 pb-2">
                  <div className="flex items-center gap-2">
                    <span className="flex h-5 w-5 items-center justify-center rounded bg-blue-600 text-[10px] font-bold text-white font-mono">
                      C
                    </span>
                    <span className="font-mono text-xs font-bold text-blue-400">
                      OpenAI CLIP ViT-B/32 (512D)
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-[11px] font-mono">
                    <span className="text-[#a69580]">{clipResults.length} items</span>
                    <span className="text-[#5c4d3a]">|</span>
                    <span className="text-blue-400">{clipMetrics.total_latency_ms || 0}ms</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                  {clipResults.map((item, idx) => (
                    <ResultCard
                      key={idx}
                      item={item}
                      rank={idx + 1}
                      modelName="clip"
                      onInspect={onInspect}
                    />
                  ))}
                </div>
              </div>

              {/* Right Column: SigLIP */}
              <div className="flex flex-col gap-3 rounded border-2 border-emerald-900/60 bg-[#251d16] p-3 shadow-[2px_2px_0_0_#000]">
                <div className="flex items-center justify-between border-b border-emerald-900/60 pb-2">
                  <div className="flex items-center gap-2">
                    <span className="flex h-5 w-5 items-center justify-center rounded bg-emerald-600 text-[10px] font-bold text-white font-mono">
                      S
                    </span>
                    <span className="font-mono text-xs font-bold text-emerald-400">
                      Google SigLIP ViT-B/16-224 (768D)
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-[11px] font-mono">
                    <span className="text-[#a69580]">{siglipResults.length} items</span>
                    <span className="text-[#5c4d3a]">|</span>
                    <span className="text-emerald-400">{siglipMetrics.total_latency_ms || 0}ms</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                  {siglipResults.map((item, idx) => (
                    <ResultCard
                      key={idx}
                      item={item}
                      rank={idx + 1}
                      modelName="siglip"
                      onInspect={onInspect}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : viewMode === 'video' ? (
          /* Video Grouped View */
          <VideoGroupGrid
            results={activeResults}
            modelName={activeModelName}
            onInspect={onInspect}
          />
        ) : (
          /* Flat Dense Grid View */
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {activeResults.map((item, idx) => (
              <ResultCard
                key={idx}
                item={item}
                rank={idx + 1}
                modelName={activeModelName}
                onInspect={onInspect}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
