import React, { useState } from 'react';
import { Eye, Clock, Film, Hash, ExternalLink } from 'lucide-react';

function formatTimestamp(sec) {
  if (sec === undefined || sec === null || isNaN(sec)) return '--:--';
  const totalSec = Math.floor(sec);
  const min = Math.floor(totalSec / 60).toString().padStart(2, '0');
  const s = (totalSec % 60).toString().padStart(2, '0');
  return `${min}:${s}`;
}

function getScoreBadgeClass(score) {
  if (score >= 0.8) return 'border-teal-600 bg-teal-950/80 text-teal-400';
  if (score >= 0.6) return 'border-amber-600 bg-amber-950/80 text-amber-400';
  return 'border-rose-600 bg-rose-950/80 text-rose-400';
}

export default function ResultCard({
  item,
  rank,
  modelName = 'clip',
  onInspect,
}) {
  const [imgError, setImgError] = useState(false);
  const [triedYt, setTriedYt] = useState(false);
  const score = item.score || 0;
  const isTop1 = rank === 1;

  // Resolve Image URL
  let imageUrl = item.image_url || '';
  if (!imageUrl && item.filename && item.filename.startsWith('COCO_')) {
    imageUrl = `http://images.cocodataset.org/val2014/${item.filename}`;
  } else if (!imageUrl && item.relative_path) {
    if (item.relative_path.includes('keyframes/')) {
      imageUrl = `/static/datasets/newsseek_keyframes/${item.relative_path}`;
    } else if (item.relative_path.startsWith('val2014/')) {
      imageUrl = `http://images.cocodataset.org/${item.relative_path}`;
    } else {
      imageUrl = `/static/datasets/${item.relative_path}`;
    }
  } else if (!imageUrl && item.video_id) {
    imageUrl = `https://img.youtube.com/vi/${item.video_id}/hqdefault.jpg`;
  }

  const handleImgError = (e) => {
    if (item.video_id && !triedYt) {
      setTriedYt(true);
      e.target.src = `https://img.youtube.com/vi/${item.video_id}/hqdefault.jpg`;
    } else {
      setImgError(true);
    }
  };

  return (
    <div
      onClick={() => onInspect(item)}
      className={`group relative flex flex-col overflow-hidden rounded border-2 transition-all cursor-pointer select-none bg-[#2a221b] ${
        isTop1
          ? 'border-[#f59e0b] shadow-[3px_3px_0_0_#000]'
          : 'border-[#5c4d3a] hover:border-[#a68a64] shadow-[2px_2px_0_0_#000] hover:translate-y-[-2px]'
      }`}
    >
      {/* Thumbnail Aspect Container */}
      <div className="relative aspect-video w-full overflow-hidden bg-[#1a1410] flex items-center justify-center">
        {imgError || !imageUrl ? (
          <div className="flex flex-col items-center justify-center p-4 text-[#6b5c4c]">
            <Film size={24} className="mb-1" />
            <span className="text-[10px] font-mono">Frame Preview</span>
          </div>
        ) : (
          <img
            src={imageUrl}
            alt={item.video_title || item.caption || `Result #${rank}`}
            loading="lazy"
            onError={handleImgError}
            className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
          />
        )}

        {/* Rank Stamp Badge */}
        <div
          className={`absolute left-2 top-2 z-10 flex h-6 w-6 items-center justify-center rounded border-2 font-mono text-xs font-bold shadow-[1px_1px_0_0_#000] ${
            isTop1
              ? 'border-[#f59e0b] bg-[#f59e0b] text-[#211a14]'
              : 'border-[#5c4d3a] bg-[#211a14] text-[#ece1cd]'
          }`}
        >
          #{rank}
        </div>

        {/* Model Tag Badge */}
        <div className="absolute right-2 top-2 z-10">
          <span
            className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider border shadow-[1px_1px_0_0_#000] ${
              modelName === 'siglip'
                ? 'border-emerald-600 bg-emerald-950/90 text-emerald-300'
                : 'border-blue-600 bg-blue-950/90 text-blue-300'
            }`}
          >
            {modelName === 'siglip' ? '768D SigLIP' : '512D CLIP'}
          </span>
        </div>

        {/* Quick Inspect Hover Overlay */}
        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-1.5 text-xs font-bold text-white font-mono">
          <Eye size={16} />
          <span>INSPECT FRAME</span>
        </div>
      </div>

      {/* Card Content & Metadata */}
      <div className="flex flex-col p-2.5 gap-2 border-t-2 border-[#5c4d3a] bg-[#211a14]">
        {/* Title or Caption */}
        <p className="line-clamp-2 text-xs font-medium text-[#ece1cd] leading-tight" title={item.video_title || item.caption}>
          {item.video_title || item.caption || `Frame #${item.frame_number || item.id}`}
        </p>

        {/* Technical Badges Row */}
        <div className="flex flex-wrap items-center justify-between gap-1 text-[11px] font-mono pt-1 border-t border-[#3d3228]">
          <div className="flex items-center gap-2 text-[#a69580]">
            {item.timestamp_sec !== undefined && (
              <span className="flex items-center gap-1">
                <Clock size={11} className="text-[#f59e0b]" />
                {formatTimestamp(item.timestamp_sec)}
              </span>
            )}
            {item.frame_number !== undefined && (
              <span className="flex items-center gap-0.5">
                <Hash size={11} />
                {item.frame_number}
              </span>
            )}
          </div>

          {/* Confidence Score Pill */}
          <div
            className={`flex items-center gap-1 rounded px-1.5 py-0.5 border text-[11px] font-bold ${getScoreBadgeClass(
              score
            )}`}
          >
            <span>{(score * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
