import React from 'react';
import { Film, Clock, ExternalLink } from 'lucide-react';

function formatTimestamp(sec) {
  if (sec === undefined || sec === null || isNaN(sec)) return '--:--';
  const totalSec = Math.floor(sec);
  const min = Math.floor(totalSec / 60).toString().padStart(2, '0');
  const s = (totalSec % 60).toString().padStart(2, '0');
  return `${min}:${s}`;
}

export default function VideoGroupGrid({
  results = [],
  modelName = 'siglip',
  onInspect,
}) {
  // Group results by video_id
  const videoGroups = {};
  results.forEach((item) => {
    const vid = item.video_id || item.id || 'single_images';
    if (!videoGroups[vid]) {
      videoGroups[vid] = {
        video_id: vid,
        video_title: item.video_title || item.caption || vid,
        language: item.language || 'en',
        max_score: item.score || 0,
        frames: [],
      };
    }
    videoGroups[vid].frames.push(item);
    if ((item.score || 0) > videoGroups[vid].max_score) {
      videoGroups[vid].max_score = item.score || 0;
    }
  });

  const sortedGroups = Object.values(videoGroups).sort(
    (a, b) => b.max_score - a.max_score
  );

  return (
    <div className="flex flex-col gap-4">
      {sortedGroups.map((group, gIdx) => {
        // Sort frames by timestamp_sec
        const sortedFrames = [...group.frames].sort(
          (a, b) => (a.timestamp_sec || 0) - (b.timestamp_sec || 0)
        );

        return (
          <div
            key={group.video_id}
            className="flex flex-col rounded border-2 border-[#5c4d3a] bg-[#2a221b] p-3 shadow-[2px_2px_0_0_#000]"
          >
            {/* Video Group Header */}
            <div className="flex flex-wrap items-center justify-between border-b border-[#3d3228] pb-2 mb-3 gap-2">
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded border border-[#f59e0b] bg-[#211a14] font-mono text-xs font-bold text-[#f59e0b]">
                  #{gIdx + 1}
                </span>
                <div>
                  <h3 className="text-xs font-bold text-[#ece1cd] line-clamp-1">
                    {group.video_title}
                  </h3>
                  <div className="flex items-center gap-2 text-[10px] font-mono text-[#a69580]">
                    <span>ID: {group.video_id}</span>
                    <span>•</span>
                    <span className="uppercase text-[#f59e0b] font-bold">
                      {group.language}
                    </span>
                    <span>•</span>
                    <span>{group.frames.length} matched keyframes</span>
                  </div>
                </div>
              </div>

              {/* Max Pooled Relevance Score */}
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-[#a69580]">
                  Max-Pool Score:
                </span>
                <span className="rounded border border-emerald-600 bg-emerald-950/80 px-2 py-0.5 font-mono text-xs font-bold text-emerald-400">
                  {(group.max_score * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Horizontal Filmstrip of Keyframes */}
            <div className="flex gap-2.5 overflow-x-auto pb-1 scrollbar-thin">
              {sortedFrames.map((kf, kIdx) => {
                let imageUrl = kf.image_url || '';
                if (!imageUrl && kf.relative_path) {
                  if (kf.relative_path.includes('keyframes/')) {
                    imageUrl = `/static/datasets/newsseek_keyframes/${kf.relative_path}`;
                  } else {
                    imageUrl = `/static/datasets/coco_test5k/${kf.relative_path}`;
                  }
                } else if (!imageUrl && kf.video_id) {
                  imageUrl = `https://img.youtube.com/vi/${kf.video_id}/hqdefault.jpg`;
                }

                return (
                  <div
                    key={kIdx}
                    onClick={() => onInspect(kf)}
                    className="group relative flex-shrink-0 w-44 rounded border border-[#5c4d3a] bg-[#211a14] overflow-hidden cursor-pointer hover:border-[#f59e0b] transition-all"
                  >
                    <div className="relative aspect-video w-full bg-[#1a1410]">
                      <img
                        src={imageUrl}
                        alt={`Frame ${kf.frame_number}`}
                        loading="lazy"
                        onError={(e) => {
                          if (kf.video_id && !e.target.dataset.fallback) {
                            e.target.dataset.fallback = 'true';
                            e.target.src = `https://img.youtube.com/vi/${kf.video_id}/hqdefault.jpg`;
                          }
                        }}
                        className="h-full w-full object-cover group-hover:scale-105 transition-transform"
                      />
                      <div className="absolute bottom-1 right-1 rounded bg-black/75 px-1 font-mono text-[9px] text-white">
                        {formatTimestamp(kf.timestamp_sec)}
                      </div>
                    </div>
                    <div className="p-1.5 flex items-center justify-between text-[10px] font-mono">
                      <span className="text-[#a69580]">F#{kf.frame_number}</span>
                      <span className="font-bold text-[#f59e0b]">
                        {(kf.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
