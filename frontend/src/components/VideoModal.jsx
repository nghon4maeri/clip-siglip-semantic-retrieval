import React, { useEffect } from 'react';
import { X, Clock, Film, Hash, Copy, Check, Search, ExternalLink } from 'lucide-react';

function formatTimestamp(sec) {
  if (sec === undefined || sec === null || isNaN(sec)) return '--:--';
  const totalSec = Math.floor(sec);
  const min = Math.floor(totalSec / 60).toString().padStart(2, '0');
  const s = (totalSec % 60).toString().padStart(2, '0');
  const ms = Math.floor((sec % 1) * 1000).toString().padStart(3, '0');
  return `${min}:${s}.${ms}`;
}

export default function VideoModal({
  item,
  onClose,
  onReverseSearch,
}) {
  const [copied, setCopied] = React.useState(false);
  const [currentImg, setCurrentImg] = React.useState('');
  const [triedYt, setTriedYt] = React.useState(false);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!item) return null;

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
    }
  };

  const copyMetadata = () => {
    const text = `Video ID: ${item.video_id || item.id}, Frame: ${item.frame_number}, Timestamp: ${item.timestamp_sec}s`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-150">
      <div 
        className="relative flex flex-col md:flex-row w-full max-w-5xl max-h-[90vh] overflow-hidden rounded-lg border-2 border-[#5c4d3a] bg-[#211a14] text-[#ece1cd] shadow-[6px_6px_0_0_#000]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute right-3 top-3 z-30 flex h-8 w-8 items-center justify-center rounded-full border-2 border-[#5c4d3a] bg-[#2a221b] text-[#ece1cd] hover:border-[#ea580c] hover:text-[#ea580c] transition-colors"
        >
          <X size={16} />
        </button>

        {/* Left: Frame Display Container */}
        <div className="flex-1 flex flex-col items-center justify-center bg-[#140f0c] p-4 border-b-2 md:border-b-0 md:border-r-2 border-[#5c4d3a] min-h-[320px]">
          <div className="relative max-h-[60vh] max-w-full overflow-hidden rounded border border-[#3d3228]">
            <img
              src={imageUrl}
              alt="Inspected Frame"
              onError={handleImgError}
              className="max-h-[58vh] w-auto object-contain"
            />
            <div className="absolute bottom-2 left-2 rounded bg-black/80 px-2 py-0.5 font-mono text-xs text-[#f59e0b] border border-[#5c4d3a]">
              {formatTimestamp(item.timestamp_sec)}
            </div>
          </div>
        </div>

        {/* Right: Technical Inspector Panel */}
        <div className="w-full md:w-80 flex flex-col p-5 gap-4 overflow-y-auto bg-[#2a221b]">
          <div>
            <span className="rounded bg-[#ea580c]/20 border border-[#ea580c] px-2 py-0.5 font-mono text-[10px] font-bold text-[#ea580c] uppercase tracking-wider">
              Frame Inspector
            </span>
            <h2 className="text-sm font-bold mt-2 text-[#ece1cd] leading-snug">
              {item.video_title || item.caption || item.id}
            </h2>
          </div>

          {/* Details Table */}
          <div className="flex flex-col gap-2 rounded border border-[#5c4d3a] bg-[#211a14] p-3 text-xs font-mono">
            <div className="flex justify-between border-b border-[#3d3228] pb-1.5">
              <span className="text-[#a69580]">Item ID:</span>
              <span className="font-bold text-[#ece1cd]">{item.id || item.video_id}</span>
            </div>
            {item.frame_number !== undefined && (
              <div className="flex justify-between border-b border-[#3d3228] pb-1.5">
                <span className="text-[#a69580]">Frame Number:</span>
                <span className="font-bold text-[#ece1cd]">#{item.frame_number}</span>
              </div>
            )}
            {item.timestamp_sec !== undefined && (
              <div className="flex justify-between border-b border-[#3d3228] pb-1.5">
                <span className="text-[#a69580]">Timestamp:</span>
                <span className="font-bold text-[#f59e0b]">{formatTimestamp(item.timestamp_sec)}</span>
              </div>
            )}
            {item.language && (
              <div className="flex justify-between border-b border-[#3d3228] pb-1.5">
                <span className="text-[#a69580]">Language:</span>
                <span className="uppercase font-bold text-[#ece1cd]">{item.language}</span>
              </div>
            )}
            <div className="flex justify-between pt-0.5">
              <span className="text-[#a69580]">Query Match Score:</span>
              <span className="font-bold text-emerald-400">
                {((item.score || 0) * 100).toFixed(2)}%
              </span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col gap-2 mt-auto pt-3 border-t border-[#3d3228]">
            {item.video_id && (
              <a
                href={`https://www.youtube.com/watch?v=${item.video_id}&t=${Math.floor(item.timestamp_sec || 0)}s`}
                target="_blank"
                rel="noreferrer"
                className="flex items-center justify-center gap-1.5 py-2 px-3 rounded border border-rose-700 bg-rose-950/70 text-rose-300 text-xs font-mono font-bold hover:bg-rose-900/80 transition-all"
              >
                <ExternalLink size={14} />
                <span>Watch Video on YouTube</span>
              </a>
            )}

            <button
              onClick={copyMetadata}
              className="flex items-center justify-center gap-1.5 py-2 px-3 rounded border border-[#5c4d3a] bg-[#211a14] text-xs font-mono font-bold hover:border-[#ece1cd] transition-all"
            >
              {copied ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              <span>{copied ? 'Metadata Copied!' : 'Copy Verification Data'}</span>
            </button>

            {onReverseSearch && imageUrl && (
              <button
                onClick={() => {
                  onReverseSearch(imageUrl);
                  onClose();
                }}
                className="btn-retro-stamp flex items-center justify-center gap-1.5 py-2 px-3 rounded bg-[#ea580c] text-white text-xs font-bold hover:bg-[#f97316]"
              >
                <Search size={14} />
                <span>Find Similar Frames</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
