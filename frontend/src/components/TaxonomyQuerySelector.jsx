import React, { useState, useEffect, useMemo } from 'react';
import { Target, Clock, Layers, Globe, Play, ChevronDown, ChevronUp, Sparkles, Search, Filter } from 'lucide-react';

export default function TaxonomyQuerySelector({ onSelectQuery, activeDataset }) {
  const [activeTab, setActiveTab] = useState('all'); // 'all', 'kis', 'temporal', 'avs'
  const [lang, setLang] = useState('vi'); // 'vi' or 'en'
  const [searchTerm, setSearchTerm] = useState('');
  const [expanded, setExpanded] = useState(true);
  const [data, setData] = useState({
    textual_kis: [],
    temporal_alignment: [],
    adhoc_video_search: [],
  });

  useEffect(() => {
    fetch('/api/taxonomy-queries')
      .then((r) => r.json())
      .then((res) => {
        if (res && (res.textual_kis || res.temporal_alignment || res.adhoc_video_search)) {
          setData(res);
        }
      })
      .catch((e) => console.error('Error fetching taxonomy queries:', e));
  }, []);

  const kisList = data.textual_kis || [];
  const temporalList = data.temporal_alignment || [];
  const avsList = data.adhoc_video_search || [];
  const totalCount = kisList.length + temporalList.length + avsList.length;

  // Filter items by search term and active tab
  const filteredKIS = useMemo(() => {
    if (!searchTerm.trim()) return kisList;
    const term = searchTerm.toLowerCase();
    return kisList.filter((it) =>
      (it.theme && it.theme.toLowerCase().includes(term)) ||
      (it.query_vi && it.query_vi.toLowerCase().includes(term)) ||
      (it.query_en && it.query_en.toLowerCase().includes(term)) ||
      (it.target_video_id && it.target_video_id.toLowerCase().includes(term))
    );
  }, [kisList, searchTerm]);

  const filteredTemporal = useMemo(() => {
    if (!searchTerm.trim()) return temporalList;
    const term = searchTerm.toLowerCase();
    return temporalList.filter((it) =>
      (it.theme && it.theme.toLowerCase().includes(term)) ||
      (it.steps_vi && it.steps_vi.some((s) => s.toLowerCase().includes(term))) ||
      (it.steps_en && it.steps_en.some((s) => s.toLowerCase().includes(term))) ||
      (it.video_id_context && it.video_id_context.toLowerCase().includes(term))
    );
  }, [temporalList, searchTerm]);

  const filteredAVS = useMemo(() => {
    if (!searchTerm.trim()) return avsList;
    const term = searchTerm.toLowerCase();
    return avsList.filter((it) =>
      (it.concept && it.concept.toLowerCase().includes(term)) ||
      (it.query_vi && it.query_vi.toLowerCase().includes(term)) ||
      (it.query_en && it.query_en.toLowerCase().includes(term))
    );
  }, [avsList, searchTerm]);

  return (
    <div className="flex flex-col gap-2 rounded border border-[#5c4d3a] bg-[#251d16] p-2.5 shadow-[2px_2px_0_0_#000]">
      {/* Header Bar */}
      <div className="flex items-center justify-between border-b border-[#3d3228] pb-2">
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1.5 font-mono text-xs font-bold text-[#ece1cd] hover:text-[#f59e0b] transition-colors"
        >
          <Sparkles size={13} className="text-[#ea580c]" />
          <span>VLM Taxonomy Bank ({totalCount})</span>
          {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>

        {/* Language Switch */}
        <div className="flex items-center gap-1 font-mono text-[10px]">
          <button
            type="button"
            onClick={() => setLang('vi')}
            className={`px-1.5 py-0.5 rounded border transition-all ${
              lang === 'vi'
                ? 'border-[#ea580c] bg-[#ea580c] text-white font-bold'
                : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
            }`}
          >
            VI
          </button>
          <button
            type="button"
            onClick={() => setLang('en')}
            className={`px-1.5 py-0.5 rounded border transition-all ${
              lang === 'en'
                ? 'border-[#3b82f6] bg-[#3b82f6] text-white font-bold'
                : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
            }`}
          >
            EN
          </button>
        </div>
      </div>

      {expanded && (
        <div className="flex flex-col gap-2">
          {/* Search Filter Inside Query Bank */}
          <div className="relative">
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Lọc truy vấn (e.g. cháy rừng, tên lửa, solar, y tế)..."
              className="w-full rounded border border-[#5c4d3a] bg-[#1a1410] px-2 py-1 text-[11px] text-[#ece1cd] placeholder-[#6b5c4c] focus:border-[#ea580c] focus:outline-none font-mono"
            />
            {searchTerm && (
              <button
                type="button"
                onClick={() => setSearchTerm('')}
                className="absolute right-2 top-1 text-[10px] text-[#a69580] hover:text-white"
              >
                ×
              </button>
            )}
          </div>

          {/* Category Tabs */}
          <div className="flex items-center gap-1 font-mono text-[9px] flex-wrap">
            <button
              type="button"
              onClick={() => setActiveTab('all')}
              className={`flex-1 py-1 px-1 rounded border text-center transition-all ${
                activeTab === 'all'
                  ? 'border-[#ece1cd] bg-[#3a3026] text-white font-bold'
                  : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
              }`}
            >
              All ({totalCount})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('kis')}
              className={`flex-1 py-1 px-1 rounded border text-center transition-all ${
                activeTab === 'kis'
                  ? 'border-[#ea580c] bg-[#ea580c]/20 text-[#ea580c] font-bold'
                  : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
              }`}
            >
              KIS ({kisList.length})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('temporal')}
              className={`flex-1 py-1 px-1 rounded border text-center transition-all ${
                activeTab === 'temporal'
                  ? 'border-[#f59e0b] bg-[#f59e0b]/20 text-[#f59e0b] font-bold'
                  : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
              }`}
            >
              Temporal ({temporalList.length})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('avs')}
              className={`flex-1 py-1 px-1 rounded border text-center transition-all ${
                activeTab === 'avs'
                  ? 'border-emerald-500 bg-emerald-950/40 text-emerald-300 font-bold'
                  : 'border-[#5c4d3a] bg-[#211a14] text-[#a69580] hover:border-[#8a755d]'
              }`}
            >
              AVS ({avsList.length})
            </button>
          </div>

          {/* Query Items List */}
          <div className="flex flex-col gap-1.5 max-h-64 overflow-y-auto pr-0.5 scrollbar-thin">
            {/* KIS section */}
            {(activeTab === 'all' || activeTab === 'kis') &&
              filteredKIS.map((item) => {
                const queryText = lang === 'vi' ? item.query_vi : item.query_en;
                return (
                  <div
                    key={item.id}
                    onClick={() => onSelectQuery(queryText, 'newsseek')}
                    className="group flex flex-col gap-1 p-2 rounded border border-[#5c4d3a] bg-[#211a14] hover:border-[#ea580c] cursor-pointer transition-all hover:translate-x-0.5"
                  >
                    <div className="flex items-center justify-between text-[10px] font-mono">
                      <span className="font-bold text-[#ea580c]">{item.id}: {item.theme}</span>
                      <span className="rounded bg-[#2a221b] border border-[#3d3228] px-1 text-[9px] text-[#a69580]">
                        {item.target_video_id}
                      </span>
                    </div>
                    <p className="text-[11px] text-[#ece1cd] line-clamp-2 leading-snug group-hover:text-white">
                      {queryText}
                    </p>
                    {item.discriminative_cues && (
                      <span className="text-[9px] font-mono text-[#a69580] line-clamp-1 italic">
                        Cues: {item.discriminative_cues}
                      </span>
                    )}
                  </div>
                );
              })}

            {/* Temporal section */}
            {(activeTab === 'all' || activeTab === 'temporal') &&
              filteredTemporal.map((item) => {
                const steps = lang === 'vi' ? item.steps_vi : item.steps_en;
                const fullChainQuery = `${steps[0]} then ${steps[1]}`;
                return (
                  <div
                    key={item.id}
                    className="flex flex-col gap-1.5 p-2 rounded border border-[#5c4d3a] bg-[#211a14] hover:border-[#f59e0b] transition-all"
                  >
                    <div className="flex items-center justify-between text-[10px] font-mono">
                      <span className="font-bold text-[#f59e0b]">{item.id}: {item.theme}</span>
                      <span className="rounded bg-[#2a221b] border border-[#3d3228] px-1 text-[9px] text-[#a69580]">
                        {item.video_id_context}
                      </span>
                    </div>

                    {/* Step 1 button */}
                    <button
                      type="button"
                      onClick={() => onSelectQuery(steps[0], 'newsseek')}
                      className="text-left p-1.5 rounded border border-[#3d3228] bg-[#1a1410] hover:border-blue-500 text-[10px] text-[#ece1cd] transition-colors"
                    >
                      <span className="font-mono font-bold text-blue-400 mr-1">E1:</span>
                      {steps[0]}
                    </button>

                    {/* Step 2 button */}
                    <button
                      type="button"
                      onClick={() => onSelectQuery(steps[1], 'newsseek')}
                      className="text-left p-1.5 rounded border border-[#3d3228] bg-[#1a1410] hover:border-emerald-500 text-[10px] text-[#ece1cd] transition-colors"
                    >
                      <span className="font-mono font-bold text-emerald-400 mr-1">E2:</span>
                      {steps[1]}
                    </button>

                    {/* Full Sequence Run */}
                    <button
                      type="button"
                      onClick={() => onSelectQuery(fullChainQuery, 'newsseek')}
                      className="flex items-center justify-center gap-1 py-1 rounded bg-[#3a3026] border border-[#f59e0b] font-mono text-[9px] font-bold text-[#f59e0b] hover:bg-[#f59e0b] hover:text-[#211a14] transition-all"
                    >
                      <Play size={9} />
                      <span>Search Combined Temporal Sequence</span>
                    </button>
                  </div>
                );
              })}

            {/* AVS section */}
            {(activeTab === 'all' || activeTab === 'avs') &&
              filteredAVS.map((item) => {
                const queryText = lang === 'vi' ? item.query_vi : item.query_en;
                return (
                  <div
                    key={item.id}
                    onClick={() => onSelectQuery(queryText, 'newsseek')}
                    className="group flex flex-col gap-1 p-2 rounded border border-[#5c4d3a] bg-[#211a14] hover:border-emerald-500 cursor-pointer transition-all hover:translate-x-0.5"
                  >
                    <div className="flex items-center justify-between text-[10px] font-mono">
                      <span className="font-bold text-emerald-400">{item.id}: {item.concept}</span>
                      <span className="rounded bg-emerald-950/80 border border-emerald-800 px-1 text-[9px] text-emerald-300">
                        Multi-Video
                      </span>
                    </div>
                    <p className="text-[11px] text-[#ece1cd] line-clamp-2 leading-snug group-hover:text-white">
                      {queryText}
                    </p>
                    {item.target_multi_videos && (
                      <span className="text-[9px] font-mono text-[#a69580] line-clamp-1">
                        Targets: {item.target_multi_videos}
                      </span>
                    )}
                  </div>
                );
              })}

            {/* Empty state if search term has no matches */}
            {filteredKIS.length === 0 && filteredTemporal.length === 0 && filteredAVS.length === 0 && (
              <div className="p-4 text-center text-xs font-mono text-[#a69580]">
                Không tìm thấy truy vấn phù hợp với từ khóa "{searchTerm}".
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
