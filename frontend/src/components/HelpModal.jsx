import React, { useEffect } from 'react';
import { X, Keyboard, Cpu, Layers } from 'lucide-react';

export default function HelpModal({ isOpen, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-150">
      <div 
        className="relative flex flex-col w-full max-w-2xl max-h-[85vh] overflow-hidden rounded-lg border-2 border-[#5c4d3a] bg-[#211a14] text-[#ece1cd] shadow-[6px_6px_0_0_#000]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b-2 border-[#5c4d3a] bg-[#2a221b] p-4">
          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded border border-[#f59e0b] bg-[#211a14] font-mono text-xs font-bold text-[#f59e0b]">
              ?
            </span>
            <h2 className="text-sm font-bold font-mono uppercase tracking-wider text-[#ece1cd]">
              System Guide & Keyboard Shortcuts
            </h2>
          </div>
          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-full border border-[#5c4d3a] hover:border-[#ea580c] hover:text-[#ea580c]"
          >
            <X size={14} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-5 text-xs font-mono">
          {/* Shortcuts */}
          <div>
            <div className="flex items-center gap-1.5 text-xs font-bold text-[#f59e0b] mb-2.5">
              <Keyboard size={14} />
              <span>KEYBOARD SHORTCUTS</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px]">
              <div className="flex items-center justify-between p-2 rounded border border-[#3d3228] bg-[#2a221b]">
                <span className="text-[#a69580]">Run Search</span>
                <kbd className="px-1.5 py-0.5 rounded border border-[#5c4d3a] bg-[#1a1410] text-[#f59e0b]">Ctrl + Enter</kbd>
              </div>
              <div className="flex items-center justify-between p-2 rounded border border-[#3d3228] bg-[#2a221b]">
                <span className="text-[#a69580]">Focus Search Input</span>
                <kbd className="px-1.5 py-0.5 rounded border border-[#5c4d3a] bg-[#1a1410] text-[#f59e0b]">/</kbd>
              </div>
              <div className="flex items-center justify-between p-2 rounded border border-[#3d3228] bg-[#2a221b]">
                <span className="text-[#a69580]">Close Active Modal</span>
                <kbd className="px-1.5 py-0.5 rounded border border-[#5c4d3a] bg-[#1a1410] text-[#f59e0b]">Esc</kbd>
              </div>
            </div>
          </div>

          {/* Model Architecture Core Comparison */}
          <div>
            <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-400 mb-2.5">
              <Cpu size={14} />
              <span>CORE ARCHITECTURAL DIFFERENCES</span>
            </div>
            <div className="flex flex-col gap-2 rounded border border-[#3d3228] bg-[#2a221b] p-3 text-[11px] font-sans leading-relaxed text-[#a69580]">
              <p>
                <strong className="text-[#ece1cd]">Pairwise Sigmoid vs InfoNCE Softmax:</strong> CLIP sử dụng Softmax chuẩn hóa mẫu số trên toàn batch, gây hiện tượng triệt tiêu gradient khi xuất hiện nhiều ảnh có nội dung tương đồng trong một batch. SigLIP đối xử mỗi cặp (ảnh, văn bản) như một bài toán phân loại nhị phân độc lập bằng Sigmoid, giúp ổn định không gian embedding.
              </p>
              <p>
                <strong className="text-[#ece1cd]">Spatial Token Grid:</strong> SigLIP phân chia ảnh thành 196 patch tokens (16x16) so với chỉ 49 patch tokens (32x32) của CLIP, cung cấp độ phân giải không gian cao gấp 4 lần để nhận diện chính xác các chi tiết nhỏ trong video và hình ảnh.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
