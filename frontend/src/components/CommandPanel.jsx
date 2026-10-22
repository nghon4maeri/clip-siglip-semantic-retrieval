import React from 'react';
import { 
  Search, Sliders, Database, Layers, ArrowRight, RotateCcw, 
  Sparkles, Image as ImageIcon, ChevronLeft, ChevronRight, HelpCircle, BarChart3
} from 'lucide-react';
import TaxonomyQuerySelector from './TaxonomyQuerySelector';

export default function CommandPanel({
  query,
  setQuery,
  dataset,
  setDataset,
  strategy,
  setStrategy,
  topK,
  setTopK,
  loading,
  onSearch,
  onImageSearch,
  presets = [],
  datasetsInfo = {},
  collapsed,
  setCollapsed,
  onOpenHelp,
  onOpenBenchmark,
}) {
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onImageSearch(file);
    }
  };

  return (
    <aside 
      className={`relative flex flex-col transition-all duration-200 border-r-2 border-[#5c4d3a] bg-[#2a221b] text-[#ece1cd] select-none ${
        collapsed ? 'w-14' : 'w-full md:w-[420px] lg:w-[460px]'
      }`}
    >
      {/* Collapse Toggle Button */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute -right-3.5 top-6 z-20 flex h-7 w-7 items-center justify-center rounded-full border-2 border-[#5c4d3a] bg-[#3a3026] text-[#ece1cd] shadow-[2px_2px_0_0_#000] hover:bg-[#4a3e32]"
        title={collapsed ? 'Mở rộng bảng điều khiển' : 'Thu gọn bảng điều khiển'}
      >
        {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>

      {collapsed ? (
        <div className="flex flex-col items-center py-6 gap-6">
          <div className="flex h-10 w-10 items-center justify-center rounded border-2 border-[#5c4d3a] bg-[#211a14] font-mono text-sm font-bold text-[#f59e0b]">
            CMD
          </div>
          <button 
            onClick={() => setCollapsed(false)}
            className="flex h-9 w-9 items-center justify-center rounded border border-[#5c4d3a] hover:bg-[#3a3026]"
          >
            <Search size={16} />
          </button>
          <button 
            onClick={() => setCollapsed(false)}
            className="flex h-9 w-9 items-center justify-center rounded border border-[#5c4d3a] hover:bg-[#3a3026]"
          >
            <Sliders size={16} />
          </button>
        </div>
      ) : (
        <div className="flex flex-col h-full overflow-y-auto p-4 gap-5">
          {/* Header Title */}
          <div className="flex items-center justify-between border-b-2 border-[#5c4d3a] pb-3">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded border border-[#f59e0b] bg-[#211a14] font-mono text-xs font-bold text-[#f59e0b]">
                &gt;_
              </div>
              <div>
                <h2 className="text-sm font-bold tracking-wider uppercase font-mono text-[#f59e0b]">
                  Command Center
                </h2>
                <p className="text-[11px] text-[#a69580]">Multimodal Retrieval Workstation</p>
              </div>
            </div>
            <button
              onClick={onOpenHelp}
              className="flex items-center gap-1 text-[11px] px-2 py-1 rounded border border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:text-[#ece1cd] hover:border-[#f59e0b]"
            >
              <HelpCircle size={13} />
              <span>Guide</span>
            </button>
          </div>

          {/* Dataset Selector */}
          <div className="flex flex-col gap-1.5">
            <label className="flex items-center gap-1.5 text-xs font-bold font-mono text-[#a69580]">
              <Database size={13} className="text-[#f59e0b]" />
              <span>ACTIVE DATASET</span>
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setDataset('coco')}
                className={`flex flex-col p-2 text-left rounded border-2 transition-all text-xs ${
                  dataset === 'coco'
                    ? 'border-[#f59e0b] bg-[#3a3026] text-[#ece1cd] font-bold shadow-[2px_2px_0_0_#000]'
                    : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
                }`}
              >
                <div className="flex items-center justify-between w-full">
                  <span className="font-mono font-bold">MS COCO 5K</span>
                  <span className="inline-block w-2 h-2 rounded-full bg-emerald-500"></span>
                </div>
                <span className="text-[10px] text-[#a69580] mt-0.5">5,000 ảnh tĩnh / 25K Captions</span>
              </button>

              <button
                type="button"
                onClick={() => setDataset('newsseek')}
                className={`flex flex-col p-2 text-left rounded border-2 transition-all text-xs ${
                  dataset === 'newsseek'
                    ? 'border-[#f59e0b] bg-[#3a3026] text-[#ece1cd] font-bold shadow-[2px_2px_0_0_#000]'
                    : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
                }`}
              >
                <div className="flex items-center justify-between w-full">
                  <span className="font-mono font-bold">NewsSeek 166K</span>
                  <span className="inline-block w-2 h-2 rounded-full bg-amber-500"></span>
                </div>
                <span className="text-[10px] text-[#a69580] mt-0.5">698 Video tin tức đa ngữ</span>
              </button>
            </div>
          </div>

          {/* Strategy / Model Architecture Selector */}
          <div className="flex flex-col gap-1.5">
            <label className="flex items-center gap-1.5 text-xs font-bold font-mono text-[#a69580]">
              <Layers size={13} className="text-[#3b82f6]" />
              <span>RETRIEVAL ARCHITECTURE</span>
            </label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="retro-input bg-[#211a14] border-2 border-[#5c4d3a] text-xs font-mono font-medium focus:border-[#f59e0b] text-[#ece1cd]"
            >
              <option value="dual">Dual Comparative Benchmark (CLIP 512D vs SigLIP 768D)</option>
              <option value="siglip">SigLIP ViT-B/16-224 (768D, SOTA)</option>
              <option value="clip">CLIP ViT-B/32 (512D, Baseline)</option>
            </select>
          </div>

          {/* Conditional Query Section: Only on NewsSeek dataset */}
          {dataset === 'newsseek' ? (
            <>
              {/* Semantic Query Input Box */}
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <label className="flex items-center gap-1.5 text-xs font-bold font-mono text-[#a69580]">
                    <Search size={13} className="text-[#ea580c]" />
                    <span>CROSS-MODAL QUERY</span>
                  </label>
                  <span className="text-[10px] font-mono text-[#a69580]">Ctrl+Enter to Run</span>
                </div>

                <div className="relative">
                  <textarea
                    rows={3}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                        e.preventDefault();
                        onSearch();
                      }
                    }}
                    placeholder="Nhập câu mô tả tự nhiên (tiếng Anh hoặc tiếng Việt), ví dụ: 'kho vàng thỏi', 'xe cứu hỏa dập lửa'..."
                    className="w-full rounded border-2 border-[#5c4d3a] bg-[#211a14] p-2.5 text-xs text-[#ece1cd] placeholder-[#6b5c4c] focus:border-[#ea580c] focus:outline-none resize-none font-sans"
                  />
                  {query && (
                    <button
                      type="button"
                      onClick={() => setQuery('')}
                      className="absolute right-2 top-2 text-[#a69580] hover:text-[#ece1cd]"
                      title="Clear query"
                    >
                      <RotateCcw size={12} />
                    </button>
                  )}
                </div>
              </div>

              {/* VLM Taxonomy Query Bank (KIS, Temporal, AVS) */}
              <TaxonomyQuerySelector
                activeDataset={dataset}
                onSelectQuery={(selectedQuery, targetDataset) => {
                  setQuery(selectedQuery);
                  if (targetDataset) setDataset(targetDataset);
                  onSearch(selectedQuery, targetDataset);
                }}
              />

              {/* Standard Presets */}
              {presets.length > 0 && (
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#a69580]">
                    Quick Examples:
                  </span>
                  <div className="flex flex-wrap gap-1 max-h-16 overflow-y-auto p-0.5 scrollbar-thin">
                    {presets.map((p, idx) => (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => {
                          setQuery(p);
                          onSearch(p, dataset);
                        }}
                        className="rounded border border-[#5c4d3a] bg-[#211a14] px-1.5 py-0.5 text-[10px] text-[#ece1cd] hover:border-[#f59e0b] hover:bg-[#3a3026] text-left transition-colors"
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Reverse Image Search Button & Top-K */}
              <div className="flex items-center gap-2 pt-1 border-t border-[#3d3228]">
                <label className="flex-1 cursor-pointer flex items-center justify-center gap-1.5 py-1.5 px-3 rounded border border-[#5c4d3a] bg-[#211a14] text-xs font-mono text-[#a69580] hover:border-[#ece1cd] hover:text-[#ece1cd] transition-all">
                  <ImageIcon size={14} className="text-[#3b82f6]" />
                  <span>Reverse Image Search</span>
                  <input 
                    type="file" 
                    accept="image/*" 
                    onChange={handleFileUpload} 
                    className="hidden" 
                  />
                </label>
                
                <div className="flex items-center gap-1 px-2.5 py-1.5 rounded border border-[#5c4d3a] bg-[#211a14] text-xs font-mono">
                  <span className="text-[#a69580]">Top-K:</span>
                  <select
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="bg-transparent text-[#f59e0b] font-bold outline-none cursor-pointer"
                  >
                    <option value={6} className="bg-[#211a14]">6</option>
                    <option value={12} className="bg-[#211a14]">12</option>
                    <option value={24} className="bg-[#211a14]">24</option>
                    <option value={48} className="bg-[#211a14]">48</option>
                  </select>
                </div>
              </div>

              {/* Primary Rubber-Stamp Search Button */}
              <button
                type="button"
                disabled={loading || !query.trim()}
                onClick={onSearch}
                className="btn-retro-stamp flex items-center justify-center gap-2 rounded bg-[#ea580c] py-2.5 px-4 text-xs font-bold text-white shadow-[3px_3px_0_0_#000] hover:bg-[#f97316] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent"></span>
                    Searching Vectors...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <Sparkles size={14} />
                    <span>EXECUTE SEARCH</span>
                    <ArrowRight size={14} />
                  </span>
                )}
              </button>
            </>
          ) : (
            /* COCO Benchmark Mode: Offline Evaluated Standardized Metrics */
            <div className="flex flex-col gap-3 rounded border-2 border-[#5c4d3a] bg-[#211a14] p-3.5 shadow-[2px_2px_0_0_#000]">
              <div className="flex items-center gap-2 border-b border-[#3d3228] pb-2">
                <BarChart3 size={14} className="text-[#f59e0b]" />
                <span className="font-mono text-xs font-bold text-[#f59e0b] uppercase">COCO 5K Benchmark Mode</span>
              </div>
              <p className="text-xs text-[#a69580] leading-relaxed">
                Tập dữ liệu MS COCO Karpathy Test (5,000 ảnh tĩnh / 25,010 chú thích văn bản) được dùng để đánh giá định lượng tự động các chỉ số Recall@K, MRR, Modality Gap và Throughput.
              </p>
              <div className="rounded border border-[#3d3228] bg-[#1a1410] p-2.5 text-[11px] font-mono flex flex-col gap-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-[#a69580]">CLIP ViT-B/32:</span>
                  <span className="text-blue-400 font-bold">R@1: 30.51% | MRR: 0.4262</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#a69580]">SigLIP ViT-B/16:</span>
                  <span className="text-emerald-400 font-bold">R@1: 47.81% (+17.3%) | MRR: 0.5908</span>
                </div>
                <div className="flex justify-between items-center pt-1 border-t border-[#3d3228]/50">
                  <span className="text-[#a69580]">Modality Gap:</span>
                  <span className="text-[#ece1cd]">CLIP: 0.8269 vs SigLIP: 1.0524</span>
                </div>
              </div>
              <button
                type="button"
                onClick={onOpenBenchmark}
                className="mt-1 flex items-center justify-center gap-2 py-2 px-3 rounded border border-[#ea580c] bg-[#ea580c] hover:bg-[#c2410c] text-white font-mono text-xs font-bold transition-all shadow-[2px_2px_0_0_#000]"
              >
                <BarChart3 size={13} />
                <span>Xem Bảng Số Liệu Benchmark COCO</span>
              </button>
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
