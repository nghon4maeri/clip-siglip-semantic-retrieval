import React, { useState, useEffect } from 'react';
import {
  Search,
  Zap,
  BarChart3,
  Database,
  Cpu,
  Layers,
  Sparkles,
  ArrowRight,
  Clock,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  ExternalLink,
  ChevronRight,
  ShieldCheck,
  Film,
  Image as ImageIcon
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

function fmtTime(seconds) {
  if (seconds === undefined || seconds === null) return '00:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export default function App() {
  const [activeTab, setActiveTab] = useState('search'); // 'search' | 'benchmark' | 'vectordb'
  const [dataset, setDataset] = useState('coco'); // 'coco' | 'newsseek'
  const [query, setQuery] = useState('A person riding a motorcycle on a countryside road');
  const [topK, setTopK] = useState(6);
  
  const [status, setStatus] = useState(null);
  const [datasetsMeta, setDatasetsMeta] = useState([]);
  const [benchmark, setBenchmark] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Initial fetch of status, datasets, and benchmark
  useEffect(() => {
    fetchStatus();
    fetchDatasets();
  }, []);

  useEffect(() => {
    fetchBenchmark(dataset);
  }, [dataset]);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      console.warn("Backend offline or booting:", err);
    }
  };

  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${API_BASE}/datasets`);
      if (res.ok) {
        const data = await res.json();
        setDatasetsMeta(data.datasets || []);
      }
    } catch (err) {
      console.warn("Error fetching datasets:", err);
    }
  };

  const fetchBenchmark = async (ds) => {
    try {
      const res = await fetch(`${API_BASE}/benchmark?dataset=${ds}`);
      if (res.ok) {
        const data = await res.json();
        setBenchmark(data);
      }
    } catch (err) {
      console.warn("Error fetching benchmark:", err);
    }
  };

  const handleSearch = async (overrideQuery = null) => {
    const q = (overrideQuery !== null ? overrideQuery : query).trim();
    if (!q) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, dataset, k: topK }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Lỗi khi thực hiện tìm kiếm');
      }

      const data = await res.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const currentDatasetMeta = datasetsMeta.find((d) => d.id === dataset);

  return (
    <div className="app-container">
      {/* Top Header Section */}
      <header className="header-section">
        <div className="title-group">
          <h1>
            <Zap className="text-blue-400" size={30} />
            Dual-Model Semantic Retrieval Engine
          </h1>
          <p>
            So sánh đối đầu trực tiếp giữa <b>OpenAI CLIP (ViT-B/32, 512D)</b> và{' '}
            <b>Google SigLIP (ViT-B/16-224, 768D)</b> trên FAISS Vector Database.
          </p>
        </div>

        <div className="header-badges">
          <div className="status-pill">
            <span className="dot-pulse"></span>
            <span>{status?.device_name || 'NVIDIA RTX 4060 CUDA'}</span>
          </div>

          <button
            onClick={() => {
              fetchStatus();
              fetchBenchmark(dataset);
            }}
            className="dataset-pill-btn flex items-center gap-2"
            title="Tải lại trạng thái"
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="tabs-nav">
        <button
          onClick={() => setActiveTab('search')}
          className={`tab-btn ${activeTab === 'search' ? 'active' : ''}`}
        >
          <Search size={18} /> Live Dual-Model Search & Query Benchmark
        </button>

        <button
          onClick={() => setActiveTab('benchmark')}
          className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
        >
          <BarChart3 size={18} /> Global Benchmark & Evaluation Dashboard
        </button>

        <button
          onClick={() => setActiveTab('vectordb')}
          className={`tab-btn ${activeTab === 'vectordb' ? 'active' : ''}`}
        >
          <Database size={18} /> Vector DB Collections & Architecture
        </button>
      </nav>

      {/* ===================================================================== */}
      {/* TAB 1: Live Search & Query Telemetry */}
      {/* ===================================================================== */}
      {activeTab === 'search' && (
        <section>
          {/* Search Card */}
          <div className="search-card">
            <div className="search-input-wrapper">
              <Search className="search-icon-left" size={20} />
              <input
                type="text"
                className="search-input"
                placeholder="Nhập câu truy vấn ngữ nghĩa (tiếng Anh hoặc tiếng Việt)..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              />
              <button
                className="btn-primary"
                onClick={() => handleSearch()}
                disabled={loading}
              >
                {loading ? <RefreshCw className="animate-spin" size={18} /> : <Zap size={18} />}
                Search & Benchmark
              </button>
            </div>

            {/* Controls: Dataset switcher & Top-K */}
            <div className="search-controls">
              <div className="flex items-center gap-3">
                <span className="text-xs uppercase font-bold text-slate-400">Dataset:</span>
                <div className="dataset-pills">
                  <button
                    className={`dataset-pill-btn ${dataset === 'coco' ? 'active' : ''}`}
                    onClick={() => {
                      setDataset('coco');
                      setResults(null);
                      setQuery('A person riding a motorcycle on a countryside road');
                    }}
                  >
                    MS COCO 5K (Image-Caption)
                  </button>
                  <button
                    className={`dataset-pill-btn ${dataset === 'newsseek' ? 'active' : ''}`}
                    onClick={() => {
                      setDataset('newsseek');
                      setResults(null);
                      setQuery('Luigi Mangione state court trial hearing');
                    }}
                  >
                    NewsSeek 166K (Video Keyframes)
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-xs uppercase font-bold text-slate-400">Top-K:</span>
                <input
                  type="range"
                  min="1"
                  max="15"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="accent-blue-500 cursor-pointer"
                />
                <span className="font-mono text-sm font-bold text-slate-200">{topK}</span>
              </div>
            </div>

            {/* Preset Query Chips */}
            {currentDatasetMeta?.presets && (
              <div className="presets-row">
                <span className="text-xs font-semibold text-slate-400 flex items-center gap-1">
                  <Sparkles size={13} className="text-amber-400" /> Presets:
                </span>
                {currentDatasetMeta.presets.map((p, idx) => (
                  <button
                    key={idx}
                    className="preset-chip"
                    onClick={() => {
                      setQuery(p);
                      handleSearch(p);
                    }}
                  >
                    {p}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Error Banner */}
          {error && (
            <div className="p-4 mb-6 rounded-xl bg-amber-950/40 border border-amber-800 text-amber-200 flex items-center gap-3 text-sm">
              <AlertTriangle size={20} className="text-amber-400 shrink-0" />
              <div>
                <b>Thông báo hệ thống:</b> {error}
                <div className="text-xs text-amber-300/80 mt-1">
                  Đảm bảo các gói FAISS vector store đã được đồng bộ về thư mục <code>storage/</code> từ Kaggle GPU.
                </div>
              </div>
            </div>
          )}

          {/* Telemetry Dashboard Grid */}
          {results && results.benchmark && (
            <div className="telemetry-grid">
              {/* Telemetry 1: Latency */}
              <div className="telemetry-card">
                <div className="telemetry-label">
                  <span>⏱️ Query Latency (Total)</span>
                  <span className="text-xs font-mono text-emerald-400">FAISS In-Memory</span>
                </div>
                <div className="telemetry-value">
                  {results.benchmark.siglip?.total_latency_ms?.toFixed(1) || '0.0'}{' '}
                  <span className="text-sm font-normal text-slate-400">ms</span>
                </div>
                <div className="telemetry-sub">
                  <span>CLIP: {results.benchmark.clip?.total_latency_ms?.toFixed(1) || 0}ms</span>
                  <span>SigLIP: {results.benchmark.siglip?.total_latency_ms?.toFixed(1) || 0}ms</span>
                </div>
              </div>

              {/* Telemetry 2: Top-1 Confidence */}
              <div className="telemetry-card">
                <div className="telemetry-label">
                  <span>🎯 Top-1 Cosine Confidence</span>
                  <span className="text-xs font-mono text-blue-400">Dot-Product</span>
                </div>
                <div className="telemetry-value text-emerald-400">
                  {results.benchmark.siglip?.top1_score?.toFixed(4) || '0.0000'}
                </div>
                <div className="telemetry-sub">
                  <span>CLIP: {results.benchmark.clip?.top1_score?.toFixed(4) || 0}</span>
                  <span>SigLIP: {results.benchmark.siglip?.top1_score?.toFixed(4) || 0}</span>
                </div>
              </div>

              {/* Telemetry 3: Separation Margin */}
              <div className="telemetry-card">
                <div className="telemetry-label">
                  <span>📈 Separation Margin (Δ)</span>
                  <span className="text-xs text-slate-400">Top1 - Distractors</span>
                </div>
                <div className="telemetry-value text-blue-400">
                  +{results.benchmark.siglip?.separation_margin?.toFixed(4) || '0.0000'}
                </div>
                <div className="telemetry-sub">
                  <span>CLIP Margin: +{results.benchmark.clip?.separation_margin?.toFixed(4) || 0}</span>
                  <span>Độ phân tách nền</span>
                </div>
              </div>

              {/* Telemetry 4: Consensus */}
              <div className="telemetry-card">
                <div className="telemetry-label">
                  <span>🤝 Model Consensus (Top-{topK})</span>
                  <span className="text-xs text-amber-400">
                    {results.benchmark.agreement?.rank_1_match ? 'MATCH' : 'DIVERGENT'}
                  </span>
                </div>
                <div className="telemetry-value text-purple-400">
                  {results.benchmark.agreement?.jaccard_overlap_pct?.toFixed(1) || '0.0'}%
                </div>
                <div className="telemetry-sub">
                  <span>Trùng {results.benchmark.agreement?.intersection_count || 0}/{topK} items</span>
                  <span>Rank-1: {results.benchmark.agreement?.rank_1_match ? 'Đồng thuận' : 'Phân kỳ'}</span>
                </div>
              </div>
            </div>
          )}

          {/* Side-by-Side Dual-Model Ranked Columns */}
          {results && (
            <div className="dual-results-grid">
              {/* CỘT TRÁI: OpenAI CLIP */}
              <div className="model-column">
                <div className="model-header">
                  <span className="model-badge-clip">
                    <Zap size={14} /> OpenAI CLIP ViT-B/32 (512D)
                  </span>
                  <span className="text-xs text-slate-400 font-mono">
                    Loss: InfoNCE Softmax | {results.benchmark?.clip?.total_latency_ms?.toFixed(1) || 0}ms
                  </span>
                </div>

                {results.clip_results?.length === 0 && (
                  <p className="text-slate-500 text-sm italic">Không có kết quả nào từ CLIP.</p>
                )}

                {results.clip_results?.map((item, idx) => {
                  const score = item.score || 0;
                  const rank = item.rank || idx + 1;
                  const cap = Array.isArray(item.captions) ? item.captions[0] : item.captions;
                  const title = item.video_title || item.filename || item.id;
                  const lang = item.language || 'en';

                  return (
                    <div key={idx} className="result-card clip">
                      <div className="card-top">
                        <span className="rank-pill clip">Rank #{rank}</span>
                        <span className="score-tag">Cosine: {score.toFixed(4)}</span>
                      </div>

                      <div className="card-body">
                        <div className="thumb-wrapper">
                          <div className="thumb-placeholder">
                            <ImageIcon size={24} />
                            <span>{item.filename || item.relative_path || 'Frame'}</span>
                          </div>
                        </div>

                        <div className="card-info">
                          <div className="item-title">{title}</div>

                          {cap && (
                            <div className="item-caption">
                              <b>Caption:</b> {cap}
                            </div>
                          )}

                          <div className="item-meta">
                            {item.timestamp_sec !== undefined && (
                              <span className="timestamp-badge">
                                ⏱️ {fmtTime(item.timestamp_sec)}
                              </span>
                            )}
                            {item.scene_index !== undefined && (
                              <span>Scene #{item.scene_index}</span>
                            )}
                            <span className="lang-tag">{lang.toUpperCase()}</span>
                            <span>ID: <code>{item.id}</code></span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* CỘT PHẢI: Google SigLIP */}
              <div className="model-column">
                <div className="model-header">
                  <span className="model-badge-siglip">
                    <Sparkles size={14} /> Google SigLIP ViT-B/16-224 (768D)
                  </span>
                  <span className="text-xs text-slate-400 font-mono">
                    Loss: Pairwise Sigmoid | {results.benchmark?.siglip?.total_latency_ms?.toFixed(1) || 0}ms
                  </span>
                </div>

                {results.siglip_results?.length === 0 && (
                  <p className="text-slate-500 text-sm italic">Không có kết quả nào từ SigLIP.</p>
                )}

                {results.siglip_results?.map((item, idx) => {
                  const score = item.score || 0;
                  const rank = item.rank || idx + 1;
                  const cap = Array.isArray(item.captions) ? item.captions[0] : item.captions;
                  const title = item.video_title || item.filename || item.id;
                  const lang = item.language || 'en';

                  return (
                    <div key={idx} className="result-card siglip">
                      <div className="card-top">
                        <span className="rank-pill siglip">Rank #{rank}</span>
                        <span className="score-tag">Cosine: {score.toFixed(4)}</span>
                      </div>

                      <div className="card-body">
                        <div className="thumb-wrapper">
                          <div className="thumb-placeholder text-emerald-400">
                            <Film size={24} />
                            <span>{item.filename || item.relative_path || 'Frame'}</span>
                          </div>
                        </div>

                        <div className="card-info">
                          <div className="item-title">{title}</div>

                          {cap && (
                            <div className="item-caption" style={{ borderLeftColor: '#10b981' }}>
                              <b>Caption:</b> {cap}
                            </div>
                          )}

                          <div className="item-meta">
                            {item.timestamp_sec !== undefined && (
                              <span className="timestamp-badge">
                                ⏱️ {fmtTime(item.timestamp_sec)}
                              </span>
                            )}
                            {item.scene_index !== undefined && (
                              <span>Scene #{item.scene_index}</span>
                            )}
                            <span className="lang-tag">{lang.toUpperCase()}</span>
                            <span>ID: <code>{item.id}</code></span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </section>
      )}

      {/* ===================================================================== */}
      {/* TAB 2: Global Benchmark & Evaluation Dashboard */}
      {/* ===================================================================== */}
      {activeTab === 'benchmark' && (
        <section>
          {/* Dataset Switcher for Benchmark */}
          <div className="flex items-center gap-3 mb-6">
            <span className="text-sm font-bold text-slate-300">Chọn Tập Đánh Giá:</span>
            <div className="dataset-pills">
              <button
                className={`dataset-pill-btn ${dataset === 'coco' ? 'active' : ''}`}
                onClick={() => setDataset('coco')}
              >
                MS COCO Karpathy Test 5K
              </button>
              <button
                className={`dataset-pill-btn ${dataset === 'newsseek' ? 'active' : ''}`}
                onClick={() => setDataset('newsseek')}
              >
                NewsSeek 166K Video Keyframes
              </button>
            </div>
          </div>

          {/* Benchmark Table */}
          <div className="benchmark-table-wrapper">
            <h3 className="text-lg font-bold text-slate-100 mb-1 flex items-center gap-2">
              <BarChart3 className="text-blue-400" size={20} />
              {benchmark?.name || 'Bảng Đo Chuẩn Quốc Tế'}
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Đánh giá định lượng toàn diện trên không gian chỉ mục vector chuẩn hóa.
            </p>

            <table className="custom-table">
              <thead>
                <tr>
                  <th>Chỉ số Đánh giá (Metrics)</th>
                  <th>OpenAI CLIP (ViT-B/32, 512D)</th>
                  <th>Google SigLIP (ViT-B/16, 768D)</th>
                  <th>Chênh lệch Đối đầu (Δ)</th>
                  <th>Mô hình dẫn đầu</th>
                </tr>
              </thead>
              <tbody>
                {benchmark?.metrics?.map((m, idx) => (
                  <tr key={idx}>
                    <td className="font-semibold text-slate-200">{m.name}</td>
                    <td className="val-clip">{m.clip}</td>
                    <td className="val-siglip">{m.siglip}</td>
                    <td>
                      <span className={`delta-tag ${m.winner === 'siglip' ? 'win-siglip' : 'win-clip'}`}>
                        {m.delta}
                      </span>
                    </td>
                    <td>
                      {m.winner === 'siglip' ? (
                        <span className="text-emerald-400 font-bold text-xs flex items-center gap-1">
                          <CheckCircle2 size={14} /> SigLIP
                        </span>
                      ) : (
                        <span className="text-blue-400 font-bold text-xs">CLIP</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Theoretical Insights */}
          <h3 className="text-lg font-bold text-slate-100 mt-6 mb-2">
            💡 Phân Tích Cơ Sở Lý Thuyết & Bản Chất Toán Học
          </h3>
          <div className="insight-grid">
            {benchmark?.insights?.map((ins, idx) => (
              <div key={idx} className="insight-card">
                <h4>
                  <ShieldCheck size={18} /> {ins.title}
                </h4>
                <p>{ins.content}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ===================================================================== */}
      {/* TAB 3: Vector DB Collections & Architecture */}
      {/* ===================================================================== */}
      {activeTab === 'vectordb' && (
        <section>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="insight-card">
              <h4 className="text-blue-400">
                <Database size={18} /> Kiến Trúc 4 Collections Độc Lập
              </h4>
              <p className="mb-4">
                Hệ thống sử dụng FAISS <code>IndexFlatIP</code> tính toán Cosine Similarity trực tiếp trên vector L2-normalized:
              </p>
              <ul className="text-sm space-y-2 text-slate-300 font-mono">
                <li>• <code>storage/coco/clip/</code>: 5,000 ảnh × 512 Float32</li>
                <li>• <code>storage/coco/siglip/</code>: 5,000 ảnh × 768 Float32</li>
                <li>• <code>storage/newsseek/clip/</code>: 166,241 frames × 512 Float32</li>
                <li>• <code>storage/newsseek/siglip/</code>: 166,241 frames × 768 Float32</li>
              </ul>
            </div>

            <div className="insight-card">
              <h4 className="text-emerald-400">
                <Cpu size={18} /> Hiệu Năng & Zero External Overhead
              </h4>
              <p className="mb-3">
                Thay vì chạy máy chủ Vector DB cồng kềnh (ChromaDB/Milvus tốn 30-50ms HTTP latency), FAISS In-Memory C++ truy vấn chỉ mất <b>1 - 3 ms</b>.
              </p>
              <p className="text-xs text-slate-400">
                Mọi vector đều liên kết 1-1 với <code>metadata.json</code> chứa toàn bộ thông tin video ID, timestamp, scene index và phụ đề.
              </p>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
