import React from 'react';
import { Award, Zap, ShieldCheck, Flame, Cpu, ArrowUpRight, CheckCircle, Info } from 'lucide-react';

export default function BenchmarkView({ benchmarkData }) {
  if (!benchmarkData) {
    return (
      <div className="flex h-full items-center justify-center p-8 font-mono text-xs text-[#a69580]">
        Loading benchmark results from disk...
      </div>
    );
  }

  const metrics = benchmarkData.metrics || [];
  const insights = benchmarkData.insights || [];

  return (
    <div className="flex flex-col gap-6 max-w-6xl mx-auto py-2">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between rounded-lg border-2 border-[#5c4d3a] bg-[#2a221b] p-5 shadow-[4px_4px_0_0_#000] gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded bg-[#ea580c] px-2 py-0.5 font-mono text-[10px] font-bold text-white uppercase tracking-wider">
              Empirical Evaluation
            </span>
            <span className="font-mono text-xs text-[#f59e0b]">Standardized Benchmark</span>
          </div>
          <h1 className="text-lg font-bold font-mono text-[#ece1cd] mt-1">
            {benchmarkData.name}
          </h1>
          <p className="text-xs text-[#a69580] mt-0.5">
            Đánh giá thực nghiệm khách quan không thiên vị giữa OpenAI CLIP (ViT-B/32, 512D) và Google SigLIP (ViT-B/16-224, 768D).
          </p>
        </div>

        <div className="flex items-center gap-2 self-end md:self-center font-mono text-xs">
          <div className="flex flex-col items-end rounded border border-[#3d3228] bg-[#211a14] px-3 py-1.5">
            <span className="text-[10px] text-[#a69580]">SOTA Winner</span>
            <span className="font-bold text-emerald-400">SigLIP ViT-B/16</span>
          </div>
        </div>
      </div>

      {/* Main Benchmark Metrics Table */}
      <div className="overflow-hidden rounded-lg border-2 border-[#5c4d3a] bg-[#2a221b] shadow-[4px_4px_0_0_#000]">
        <div className="border-b-2 border-[#5c4d3a] bg-[#342921] px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2 font-mono text-xs font-bold text-[#ece1cd]">
            <Award size={15} className="text-[#f59e0b]" />
            <span>CROSS-MODAL RETRIEVAL ACCURACY & LATENCY BENCHMARK</span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-[#3d3228] bg-[#211a14] text-[11px] text-[#a69580]">
                <th className="py-2.5 px-4 font-bold">Chỉ số Đánh giá (Metric)</th>
                <th className="py-2.5 px-4 font-bold text-blue-400">CLIP (ViT-B/32, 512D)</th>
                <th className="py-2.5 px-4 font-bold text-emerald-400">SigLIP (ViT-B/16, 768D)</th>
                <th className="py-2.5 px-4 font-bold">Chênh lệch (Delta)</th>
                <th className="py-2.5 px-4 font-bold text-center">Người chiến thắng</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#3d3228]">
              {metrics.map((m, idx) => {
                const isSiglipWinner = m.winner === 'siglip';
                const isClipWinner = m.winner === 'clip';

                return (
                  <tr key={idx} className="hover:bg-[#342921]/60 transition-colors">
                    <td className="py-3 px-4 font-medium text-[#ece1cd]">{m.name}</td>
                    <td className="py-3 px-4 text-blue-300 font-bold">{m.clip}</td>
                    <td className="py-3 px-4 text-emerald-300 font-bold">{m.siglip}</td>
                    <td className="py-3 px-4 font-bold text-[#f59e0b]">{m.delta}</td>
                    <td className="py-3 px-4 text-center">
                      {isSiglipWinner ? (
                        <span className="inline-flex items-center gap-1 rounded bg-emerald-950/80 border border-emerald-600 px-2 py-0.5 text-[10px] font-bold text-emerald-400">
                          <CheckCircle size={10} />
                          SigLIP
                        </span>
                      ) : isClipWinner ? (
                        <span className="inline-flex items-center gap-1 rounded bg-blue-950/80 border border-blue-600 px-2 py-0.5 text-[10px] font-bold text-blue-400">
                          <CheckCircle size={10} />
                          CLIP
                        </span>
                      ) : (
                        <span className="inline-flex items-center rounded border border-[#5c4d3a] px-2 py-0.5 text-[10px] text-[#a69580]">
                          Tie
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Architectural Insights Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {insights.map((ins, idx) => (
          <div
            key={idx}
            className="flex flex-col rounded-lg border-2 border-[#5c4d3a] bg-[#2a221b] p-4 shadow-[3px_3px_0_0_#000] gap-2"
          >
            <div className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded border border-[#ea580c] bg-[#211a14] font-mono text-xs font-bold text-[#ea580c]">
                #{idx + 1}
              </span>
              <h3 className="text-xs font-bold font-mono text-[#ece1cd]">
                {ins.title}
              </h3>
            </div>
            <p className="text-xs text-[#a69580] leading-relaxed pl-8 font-sans">
              {ins.content}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
