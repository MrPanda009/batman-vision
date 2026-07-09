'use client';

import { useState, useEffect, useRef } from 'react';

interface ObjectItem {
  id: number;
  tags: string[];
  ocr_text: string;
  status: 'pending' | 'tagged' | 'failed';
  first_seen: number;
  last_seen: number;
  crop_paths: string[];
}

interface Stats {
  pending: number;
  tagged: number;
  failed: number;
}

const BACKEND_URL = 'http://localhost:8000';

export default function Home() {
  const [objects, setObjects] = useState<ObjectItem[]>([]);
  const [stats, setStats] = useState<Stats>({ pending: 0, tagged: 0, failed: 0 });
  const [isPipelineActive, setIsPipelineActive] = useState<boolean>(false);
  const [feedToken, setFeedToken] = useState<number>(Date.now());
  const [statusMessage, setStatusMessage] = useState<{ text: string; isError: boolean } | null>(null);
  const [confirmClear, setConfirmClear] = useState<boolean>(false);
  const [selectedObject, setSelectedObject] = useState<ObjectItem | null>(null);

  // Poll status, stats, and objects
  const fetchData = async () => {
    try {
      // 1. Fetch Pipeline Status
      const statusRes = await fetch(`${BACKEND_URL}/pipeline/status`);
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        setIsPipelineActive(statusData.active);
      }

      // 2. Fetch Stats
      const statsRes = await fetch(`${BACKEND_URL}/api/stats`);
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }

      // 3. Fetch Objects
      const objectsRes = await fetch(`${BACKEND_URL}/api/objects`);
      if (objectsRes.ok) {
        const objectsData = await objectsRes.json();
        setObjects(objectsData);
      }
    } catch (err) {
      console.error('Error fetching data from backend:', err);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchData();

    // Polling interval
    const interval = setInterval(fetchData, 1500);
    return () => clearInterval(interval);
  }, []);

  const handleStartPipeline = async () => {
    setStatusMessage(null);
    try {
      const res = await fetch(`${BACKEND_URL}/pipeline/start`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setIsPipelineActive(true);
        setFeedToken(Date.now());
        setStatusMessage({ text: data.message || 'Pipeline started successfully.', isError: false });
      } else {
        setStatusMessage({ text: data.detail || 'Failed to start pipeline.', isError: true });
      }
    } catch (err) {
      setStatusMessage({ text: 'Error connecting to backend server.', isError: true });
    }
  };

  const handleStopPipeline = async () => {
    setStatusMessage(null);
    try {
      const res = await fetch(`${BACKEND_URL}/pipeline/stop`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setIsPipelineActive(false);
        setStatusMessage({ text: data.message || 'Pipeline stopped.', isError: false });
      } else {
        setStatusMessage({ text: data.detail || 'Failed to stop pipeline.', isError: true });
      }
    } catch (err) {
      setStatusMessage({ text: 'Error connecting to backend server.', isError: true });
    }
  };

  const handleClearDatabase = async () => {
    setStatusMessage(null);
    if (isPipelineActive) {
      setStatusMessage({ text: 'Cannot clear database while pipeline is active.', isError: true });
      return;
    }

    try {
      const res = await fetch(`${BACKEND_URL}/api/clear`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setConfirmClear(false);
        setObjects([]);
        setStats({ pending: 0, tagged: 0, failed: 0 });
        setStatusMessage({ 
          text: `Cleared ${data.cleared_objects} database objects and ${data.deleted_files} crop files.`, 
          isError: false 
        });
      } else {
        setStatusMessage({ text: data.detail || 'Failed to clear database.', isError: true });
      }
    } catch (err) {
      setStatusMessage({ text: 'Error connecting to backend server.', isError: true });
    }
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const getDuration = (first: number, last: number) => {
    const duration = last - first;
    return duration > 0.1 ? `${duration.toFixed(1)}s` : '< 0.1s';
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans antialiased">
      {/* Header */}
      <header className="border-b border-zinc-900 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-cyan-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <span className="font-mono text-zinc-950 font-bold text-lg">B</span>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 font-mono">
                BATMAN VISION ORCHESTRATOR
              </h1>
              <p className="text-xs text-zinc-500 font-mono">LIVE VEHICLE / OBJECT TRACKER & TAGGER</p>
            </div>
          </div>

          {/* System Status Pill */}
          <div className="flex items-center gap-2 bg-zinc-900 px-3 py-1.5 rounded-full border border-zinc-800">
            <span className={`relative flex h-2 w-2`}>
              {isPipelineActive && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              )}
              <span className={`relative inline-flex rounded-full h-2 w-2 ${isPipelineActive ? 'bg-emerald-500' : 'bg-zinc-600'}`}></span>
            </span>
            <span className="text-xs font-medium font-mono text-zinc-300">
              {isPipelineActive ? 'SYSTEM RUNNING' : 'SYSTEM IDLE'}
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Status Notification */}
        {statusMessage && (
          <div className={`mb-6 p-4 rounded-lg border text-sm font-mono flex justify-between items-center transition-all ${
            statusMessage.isError 
              ? 'bg-red-950/30 border-red-900/50 text-red-400' 
              : 'bg-cyan-950/30 border-cyan-900/50 text-cyan-400'
          }`}>
            <span>{statusMessage.text}</span>
            <button 
              onClick={() => setStatusMessage(null)}
              className="text-xs uppercase hover:underline opacity-80"
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* LEFT PANEL: Feed + Controls + Stats */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* Live Video Feed Card */}
            <div className="bg-zinc-900 rounded-xl border border-zinc-800 overflow-hidden shadow-2xl">
              <div className="px-4 py-3 border-b border-zinc-800 bg-zinc-900/50 flex justify-between items-center">
                <h3 className="text-sm font-semibold font-mono text-zinc-200">LIVE FEED HUD</h3>
                {isPipelineActive && (
                  <span className="text-xs bg-cyan-950 text-cyan-400 px-2 py-0.5 rounded font-mono border border-cyan-800 animate-pulse">
                    RECEIVING
                  </span>
                )}
              </div>
              
              <div className="relative aspect-video bg-black flex items-center justify-center">
                {isPipelineActive ? (
                  <img
                    src={`${BACKEND_URL}/video_feed?t=${feedToken}`}
                    alt="HUD feed stream"
                    className="w-full h-full object-contain"
                    onError={() => {
                      // fallback logic on connection error
                      console.log("Feed image error, reloading...");
                    }}
                  />
                ) : (
                  <div className="text-center p-6 space-y-3 font-mono">
                    <svg className="w-12 h-12 text-zinc-700 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    <p className="text-sm text-zinc-500 font-semibold">WEBCAM DISCONNECTED</p>
                    <p className="text-xs text-zinc-600">Activate pipeline to start camera capture.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Controls Card */}
            <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5 space-y-4 shadow-xl">
              <h3 className="text-sm font-semibold font-mono text-zinc-200">PIPELINE CONTROL</h3>
              
              <div className="grid grid-cols-2 gap-4">
                <button
                  onClick={handleStartPipeline}
                  disabled={isPipelineActive}
                  className={`py-3 px-4 rounded-lg font-mono font-bold text-sm tracking-wide transition-all shadow-md ${
                    isPipelineActive 
                      ? 'bg-zinc-850 text-zinc-600 border border-zinc-800 cursor-not-allowed' 
                      : 'bg-emerald-500 text-zinc-950 hover:bg-emerald-400 hover:shadow-emerald-500/20 active:scale-95'
                  }`}
                >
                  START PIPELINE
                </button>

                <button
                  onClick={handleStopPipeline}
                  disabled={!isPipelineActive}
                  className={`py-3 px-4 rounded-lg font-mono font-bold text-sm tracking-wide transition-all shadow-md ${
                    !isPipelineActive 
                      ? 'bg-zinc-850 text-zinc-600 border border-zinc-800 cursor-not-allowed' 
                      : 'bg-rose-500 text-zinc-950 hover:bg-rose-400 hover:shadow-rose-500/20 active:scale-95'
                  }`}
                >
                  STOP PIPELINE
                </button>
              </div>

              {/* Clear Database Control */}
              <div className="border-t border-zinc-800 pt-4 mt-2">
                {!confirmClear ? (
                  <button
                    onClick={() => setConfirmClear(true)}
                    disabled={isPipelineActive}
                    className={`w-full py-2.5 px-4 rounded-lg font-mono text-xs font-semibold tracking-wide border transition-all ${
                      isPipelineActive 
                        ? 'bg-zinc-900 border-zinc-850 text-zinc-600 cursor-not-allowed' 
                        : 'bg-zinc-950 border-zinc-800 hover:border-rose-900/50 text-zinc-400 hover:text-rose-400'
                    }`}
                  >
                    CLEAR DATABASE & CROPS
                  </button>
                ) : (
                  <div className="bg-rose-950/20 border border-rose-900/40 rounded-lg p-3 space-y-3 font-mono">
                    <p className="text-xs text-rose-400 text-center font-semibold">
                      This will delete all SQLite rows and JPEG crop files. Active pipeline must remain off.
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={handleClearDatabase}
                        className="py-1.5 px-3 bg-rose-500 text-zinc-950 text-xs font-bold rounded hover:bg-rose-400"
                      >
                        CONFIRM CLEAR
                      </button>
                      <button
                        onClick={() => setConfirmClear(false)}
                        className="py-1.5 px-3 bg-zinc-800 text-zinc-300 text-xs font-semibold rounded hover:bg-zinc-700"
                      >
                        CANCEL
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Stats Card */}
            <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-5 space-y-4 shadow-xl">
              <h3 className="text-sm font-semibold font-mono text-zinc-200">REALTIME PIPELINE STATS</h3>
              
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-zinc-950 border border-zinc-800/80 p-3 rounded-lg text-center font-mono">
                  <p className="text-xs text-zinc-500 uppercase">Tagged</p>
                  <p className="text-2xl font-bold text-emerald-400 mt-1">{stats.tagged}</p>
                </div>
                
                <div className="bg-zinc-950 border border-zinc-800/80 p-3 rounded-lg text-center font-mono">
                  <p className="text-xs text-zinc-500 uppercase">Pending</p>
                  <p className="text-2xl font-bold text-amber-400 mt-1 animate-pulse">{stats.pending}</p>
                </div>

                <div className="bg-zinc-950 border border-zinc-800/80 p-3 rounded-lg text-center font-mono">
                  <p className="text-xs text-zinc-500 uppercase">Failed</p>
                  <p className="text-2xl font-bold text-rose-500 mt-1">{stats.failed}</p>
                </div>
              </div>
            </div>

          </div>

          {/* RIGHT PANEL: Gallery */}
          <div className="lg:col-span-7 space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-lg font-bold font-mono tracking-wide text-zinc-200">OBJECT DETECTIONS ({objects.length})</h2>
              <span className="text-xs text-zinc-500 font-mono">Auto-refresh active</span>
            </div>

            {objects.length === 0 ? (
              <div className="bg-zinc-900 border border-dashed border-zinc-800 rounded-xl p-12 text-center font-mono">
                <svg className="w-16 h-16 text-zinc-800 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
                <p className="text-sm font-semibold text-zinc-500">No objects tracked yet.</p>
                <p className="text-xs text-zinc-600 mt-1">Start the pipeline. Detections will automatically populate here once finalized.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[75vh] overflow-y-auto pr-2 custom-scrollbar">
                {objects.map((obj) => (
                  <div 
                    key={obj.id} 
                    className="bg-zinc-900 border border-zinc-800 hover:border-zinc-700/80 rounded-xl overflow-hidden shadow-md hover:shadow-lg transition-all group flex flex-col"
                  >
                    {/* Object Image Crop Container */}
                    <div className="relative aspect-video bg-zinc-950 overflow-hidden flex items-center justify-center cursor-pointer"
                         onClick={() => setSelectedObject(obj)}>
                      {obj.crop_paths && obj.crop_paths.length > 0 ? (
                        <img 
                          src={`${BACKEND_URL}/${obj.crop_paths[0]}`}
                          alt={`Track ${obj.id}`}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        />
                      ) : (
                        <div className="text-xs text-zinc-600 font-mono">No Crops</div>
                      )}
                      
                      {/* Top Badges */}
                      <div className="absolute top-2 left-2 flex gap-1">
                        <span className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase tracking-wider ${
                          obj.status === 'tagged' 
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-900/55' 
                            : obj.status === 'pending'
                            ? 'bg-amber-950 text-amber-400 border border-amber-900/55 animate-pulse'
                            : 'bg-rose-950 text-rose-400 border border-rose-900/55'
                        }`}>
                          {obj.status}
                        </span>
                      </div>
                      
                      {obj.crop_paths && obj.crop_paths.length > 1 && (
                        <div className="absolute bottom-2 right-2 bg-zinc-950/80 backdrop-blur border border-zinc-800 text-[10px] font-mono px-1.5 py-0.5 rounded text-zinc-300">
                          {obj.crop_paths.length} crops
                        </div>
                      )}
                    </div>

                    {/* Object Info Content */}
                    <div className="p-4 space-y-3 flex-1 flex flex-col justify-between">
                      <div className="space-y-2">
                        {/* ID and Duration */}
                        <div className="flex justify-between items-center text-xs font-mono text-zinc-500">
                          <span>Track ID #{obj.id}</span>
                          <span>Dur: {getDuration(obj.first_seen, obj.last_seen)}</span>
                        </div>

                        {/* OCR Text */}
                        {obj.ocr_text && (
                          <div className="bg-zinc-950 border border-zinc-800 px-2.5 py-1.5 rounded text-xs font-mono text-cyan-400 overflow-x-auto truncate">
                            <span className="text-[10px] text-zinc-500 uppercase block mb-0.5">Visible Text (OCR)</span>
                            "{obj.ocr_text}"
                          </div>
                        )}

                        {/* Tags */}
                        {obj.tags && obj.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1 pt-1">
                            {obj.tags.map((tag, tIdx) => (
                              <span 
                                key={tIdx} 
                                className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                                  tIdx === 0 && obj.status === 'tagged'
                                    ? 'bg-cyan-950/40 text-cyan-400 border-cyan-900/60 font-semibold' 
                                    : 'bg-zinc-950 text-zinc-400 border-zinc-800/80'
                                }`}
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Timestamps Footer */}
                      <div className="border-t border-zinc-850 pt-2 mt-2 flex justify-between text-[9px] font-mono text-zinc-600">
                        <span>Seen: {formatTime(obj.first_seen)}</span>
                        <span>Last: {formatTime(obj.last_seen)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* MULTIPLE CROPS PREVIEW MODAL */}
      {selectedObject && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fade-in">
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl max-w-2xl w-full overflow-hidden shadow-2xl">
            <div className="px-6 py-4 border-b border-zinc-800 flex justify-between items-center">
              <h3 className="font-bold font-mono text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
                Track #{selectedObject.id} — All Diverse Crops
              </h3>
              <button 
                onClick={() => setSelectedObject(null)}
                className="text-zinc-500 hover:text-zinc-300 font-mono text-sm uppercase"
              >
                Close
              </button>
            </div>
            
            <div className="p-6">
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 max-h-[50vh] overflow-y-auto pr-1">
                {selectedObject.crop_paths.map((path, idx) => (
                  <div key={idx} className="bg-zinc-950 border border-zinc-800 rounded-lg overflow-hidden relative group">
                    <img 
                      src={`${BACKEND_URL}/${path}`} 
                      alt={`Crop ${idx}`} 
                      className="w-full aspect-video object-cover"
                    />
                    <div className="absolute bottom-1 left-1 bg-black/75 px-1.5 py-0.5 rounded text-[8px] font-mono text-zinc-400">
                      Crop #{idx + 1}
                    </div>
                  </div>
                ))}
              </div>

              {/* Object Details summary inside modal */}
              <div className="mt-6 pt-4 border-t border-zinc-800 space-y-2 text-xs font-mono">
                <div>
                  <span className="text-zinc-500">Tags: </span>
                  <span className="text-zinc-300">{selectedObject.tags?.join(', ') || 'None'}</span>
                </div>
                {selectedObject.ocr_text && (
                  <div>
                    <span className="text-zinc-500">OCR Text: </span>
                    <span className="text-cyan-400">"{selectedObject.ocr_text}"</span>
                  </div>
                )}
                <div className="flex justify-between text-[10px] text-zinc-500">
                  <span>Tracked from {formatTime(selectedObject.first_seen)} to {formatTime(selectedObject.last_seen)}</span>
                  <span>Total duration: {getDuration(selectedObject.first_seen, selectedObject.last_seen)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
