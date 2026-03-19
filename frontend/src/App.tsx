import { useState, useEffect, useRef } from 'react'
import { Camera, Layers, Box, Cpu, Play, Square, Activity, Upload, RefreshCw, Database } from 'lucide-react'
import { io, Socket } from 'socket.io-client'

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'pipeline'>('dashboard');

  // Dashboard Status
  const [isConnected, setIsConnected] = useState(false);
  const [streamActive, setStreamActive] = useState(false);
  const [detections, setDetections] = useState<any[]>([]);
  const [fps, setFps] = useState(0);
  const [events, setEvents] = useState<string[]>(['[SYSTEM] Pipeline ready.', '[SYSTEM] Awaiting stream start.']);

  // Pipeline Status
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [productName, setProductName] = useState<string>('');
  const [trainEpochs, setTrainEpochs] = useState<number>(10);
  const [pipelineState, setPipelineState] = useState<'idle' | 'uploading' | 'extracting' | 'annotating' | 'training' | 'done'>('idle');
  const [pipelineLog, setPipelineLog] = useState<string>('');

  const runPipeline = async () => {
    if (!videoFile) return;
    setPipelineState('uploading');
    setPipelineLog('Uploading video...');
    const formData = new FormData();
    formData.append('file', videoFile);

    try {
      const resUpload = await fetch('http://localhost:5000/api/upload', { method: 'POST', body: formData });
      if (!resUpload.ok) throw new Error("Upload Failed");

      setPipelineLog('Extracting frames...');
      setPipelineState('extracting');
      await fetch('http://localhost:5000/api/extract', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoFile.name, target_frames: 180 })
      });

      setPipelineLog('Running auto-annotation...');
      setPipelineState('annotating');
      await fetch('http://localhost:5000/api/annotate/auto', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoFile.name, class_id: 0 })
      });

      setPipelineLog(`Training YOLOv8 (${trainEpochs} Epochs)...`);
      setPipelineState('training');
      const resTrain = await fetch('http://localhost:5000/api/train', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoFile.name, product_name: productName || 'product', epochs: trainEpochs })
      });
      const dataTrain = await resTrain.json();
      if (!resTrain.ok || !dataTrain.success) throw new Error(dataTrain.message || 'Training Failed');

      setPipelineLog('Pipeline Completed! Model saved. Go to Dashboard and load the Custom Model.');
      setPipelineState('done');
    } catch (e) {
      setPipelineLog(`Error: ${e}`);
      setPipelineState('idle');
    }
  };



  // Model Status
  const [activeModel, setActiveModel] = useState<'default' | 'custom' | null>(null);
  const [modelLoading, setModelLoading] = useState<'default' | 'custom' | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<Socket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const requestRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const lastTimeRef = useRef<number>(performance.now());
  const framesRef = useRef<number>(0);

  const addEvent = (msg: string) => {
    setEvents(prev => [...prev.slice(-19), msg]);
  };

  useEffect(() => {
    socketRef.current = io('http://localhost:5000');
    socketRef.current.on('connect', () => { setIsConnected(true); addEvent('[NETWORK] Connected to Inference Engine API.'); });
    socketRef.current.on('disconnect', () => { setIsConnected(false); addEvent('[NETWORK] Disconnected from API.'); });
    socketRef.current.on('status', (data) => addEvent(`[SYSTEM] ${data.data}`));
    socketRef.current.on('detections', (data) => {
      if (data.error) {
        if (!data.error.includes('No model loaded')) addEvent(`[ERROR] ${data.error}`);
        setDetections([]);
      }
      else if (data.boxes) setDetections(data.boxes);

      framesRef.current += 1;
      const now = performance.now();
      if (now - lastTimeRef.current >= 1000) {
        setFps(Math.round((framesRef.current * 1000) / (now - lastTimeRef.current)));
        framesRef.current = 0;
        lastTimeRef.current = now;
      }
    });
    return () => { socketRef.current?.disconnect(); };
  }, []);

  useEffect(() => {
    if (activeTab !== 'dashboard') {
      if (streamActive) setStreamActive(false);
      return;
    }
    const startWebcam = async () => {
      try {
        // Request high-resolution natively or the best the camera can do
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          }
        });
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        addEvent('[HARDWARE] Camera initialized.');

        const sendFrame = () => {
          if (!streamRef.current || !videoRef.current || !socketRef.current) return;
          const hiddenCanvas = document.createElement('canvas');
          // Scale it down slightly for the network stream to prevent latency or out of memory
          // 854x480 is 480p 16:9, which is a good sweet spot for live YOLO inference 
          const targetWidth = 854;
          const targetHeight = 480;

          hiddenCanvas.width = targetWidth;
          hiddenCanvas.height = targetHeight;
          const ctx = hiddenCanvas.getContext('2d');

          if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0, targetWidth, targetHeight);
            const base64Image = hiddenCanvas.toDataURL('image/jpeg', 0.6);
            socketRef.current.emit('stream_frame', { image: base64Image });
          }
          requestRef.current = setTimeout(() => { if (streamRef.current) sendFrame(); }, 1000 / 15);
        };
        setTimeout(() => sendFrame(), 1500);
      } catch (err) {
        addEvent(`[ERROR] Camera access denied.`);
        setStreamActive(false);
      }
    };

    const stopWebcam = () => {
      if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
      if (requestRef.current) clearTimeout(requestRef.current);
      addEvent('[HARDWARE] Camera stopped.');
      setDetections([]);
      setFps(0);
    };

    if (streamActive) startWebcam();
    else stopWebcam();

    return stopWebcam;
  }, [streamActive, activeTab]);

  useEffect(() => {
    if (activeTab !== 'dashboard' || !canvasRef.current || !videoRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx || !videoRef.current || !canvasRef.current) return;

    // Clear canvas every frame so old boxes don't get stuck
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // Stop drawing if there are no detections
    if (detections.length === 0) return;

    const videoElement = videoRef.current;
    if (!videoElement.videoWidth) return;

    // Get the actual display size of the canvas/video container
    const rect = canvasRef.current.getBoundingClientRect();
    canvasRef.current.width = rect.width;
    canvasRef.current.height = rect.height;

    // Calculate dimensions of the video intrinsically
    const vWidth = videoElement.videoWidth;
    const vHeight = videoElement.videoHeight;

    // Calculate the scale to fill the container (object-cover logic)
    const scale = Math.max(rect.width / vWidth, rect.height / vHeight);

    // Calculate the actual displayed width and height of the video
    const displayedWidth = vWidth * scale;
    const displayedHeight = vHeight * scale;

    // Calculate the padding (letterboxing) to center the video
    const offsetX = (rect.width - displayedWidth) / 2;
    const offsetY = (rect.height - displayedHeight) / 2;

    detections.forEach(box => {
      const { x1, y1, x2, y2, confidence, class: className } = box;

      // Scale received coordinates (854x480 compressed frame) back up to original camera resolution (e.g. 1080p)
      const origX1 = x1 * (vWidth / 854);
      const origY1 = y1 * (vHeight / 480);
      const origX2 = x2 * (vWidth / 854);
      const origY2 = y2 * (vHeight / 480);

      // Apply CSS scale multiplier and add letterbox offsets
      const rx = (origX1 * scale) + offsetX;
      const ry = (origY1 * scale) + offsetY;
      const rw = (origX2 - origX1) * scale;
      const rh = (origY2 - origY1) * scale;

      ctx.strokeStyle = '#00CCFF'; ctx.lineWidth = 3; ctx.strokeRect(rx, ry, rw, rh);
      ctx.fillStyle = '#00CCFF';
      const label = `${className} ${(confidence * 100).toFixed(1)}%`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(rx, ry - 20, textWidth + 10, 20);
      ctx.fillStyle = '#0F172A'; ctx.font = 'bold 12px Inter'; ctx.fillText(label, rx + 5, ry - 6);
    });
  }, [detections, activeTab]);


  return (
    <div className="h-screen overflow-hidden p-6 flex flex-col gap-6 bg-dashboard-bg">
      {/* Header */}
      <header className="flex justify-between items-center hud-container py-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-bmw-cyan flex items-center justify-center">
            <Layers className="text-dashboard-bg w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-wider uppercase text-dashboard-text">Vision<span className="text-bmw-cyan">Detect</span></h1>
            <p className="text-xs text-dashboard-muted font-mono tracking-widest uppercase">Real-time Product Inspection</p>
          </div>
        </div>

        <div className="flex bg-dashboard-bg rounded-lg p-1 border border-dashboard-border">
          <button
            className={`px-4 py-1.5 rounded-md text-sm font-semibold transition-colors ${activeTab === 'dashboard' ? 'bg-dashboard-border text-white' : 'text-dashboard-muted hover:text-dashboard-text'}`}
            onClick={() => setActiveTab('dashboard')}
          >
            Dashboard
          </button>
          <button
            className={`px-4 py-1.5 rounded-md text-sm font-semibold transition-colors ${activeTab === 'pipeline' ? 'bg-dashboard-border text-white' : 'text-dashboard-muted hover:text-dashboard-text'}`}
            onClick={() => setActiveTab('pipeline')}
          >
            Pipeline
          </button>
        </div>

        <div className="flex gap-4 items-center">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 text-xs font-mono">
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-bmw-red'}`}></span>
              Backend API
            </div>
            <div className="flex items-center gap-2 text-xs font-mono">
              <span className="w-2 h-2 rounded-full bg-bmw-cyan"></span>
              AI Engine: YOLOv8
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      {activeTab === 'dashboard' ? (
        <main className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 min-h-0 overflow-hidden">
          {/* Dashboard Left Panel */}
          <div className="col-span-1 flex flex-col gap-4">
            <div className="hud-container-accent flex-1 flex flex-col gap-4">
              <div className="flex items-center gap-2 text-bmw-cyan border-b border-dashboard-border pb-2">
                <Activity className="w-5 h-5" />
                <h2 className="font-semibold uppercase tracking-wide text-sm">Controls</h2>
              </div>
              <div className="flex flex-col gap-3 mt-2">
                <button
                  className={`py-3 rounded-lg font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-all shadow-lg ${streamActive ? 'bg-bmw-red hover:bg-red-600 text-white' : 'bg-bmw-cyan hover:bg-cyan-500 text-dashboard-bg'}`}
                  onClick={() => setStreamActive(!streamActive)}
                >
                  {streamActive ? <Square className="w-5 h-5" /> : <Play className="w-5 h-5" fill="currentColor" />}
                  {streamActive ? 'Stop Stream' : 'Start Stream'}
                </button>
                <button
                  className={`btn-secondary uppercase text-xs tracking-wider ${activeModel === 'default' ? 'bg-dashboard-border text-white border-bmw-cyan border' : ''} ${modelLoading !== null ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={modelLoading !== null}
                  onClick={() => {
                    setModelLoading('default');
                    fetch('http://localhost:5000/api/model/load', {
                      method: 'POST', headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ model_name: 'yolov8n.pt' })
                    }).then(r => r.json()).then(d => { addEvent(`[MODEL] ${d.message}`); setActiveModel('default'); setModelLoading(null); })
                      .catch(() => { addEvent(`[ERROR] Model load failed.`); setModelLoading(null); });
                  }}
                >
                  <Cpu className="w-4 h-4 text-bmw-cyan" />
                  {modelLoading === 'default' ? 'Loading Default...' : modelLoading === 'custom' ? 'Turning Off Default...' : activeModel === 'default' ? 'Running Default Model' : 'Load Default Model'}
                </button>
                <button
                  className={`btn-secondary uppercase text-xs tracking-wider border ${activeModel === 'custom' ? 'bg-dashboard-border text-white border-bmw-red' : 'border-transparent'} ${modelLoading !== null ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={modelLoading !== null}
                  onClick={() => {
                    setModelLoading('custom');
                    fetch('http://localhost:5000/api/model/load', {
                      method: 'POST', headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ model_name: 'custom_model/weights/best.pt' })
                    }).then(r => r.json()).then(d => { addEvent(d.success ? `[MODEL] Custom Model Loaded!` : `[ERROR] ${d.message}`); if (d.success) setActiveModel('custom'); setModelLoading(null); })
                      .catch(() => { addEvent(`[ERROR] Custom Model load failed.`); setModelLoading(null); });
                  }}
                >
                  <Box className="w-4 h-4 text-bmw-red" />
                  {modelLoading === 'custom' ? 'Loading Custom...' : modelLoading === 'default' ? 'Turning Off Custom...' : activeModel === 'custom' ? 'Running Custom Model' : 'Load Custom Model'}
                </button>
              </div>
              <div className="mt-6 flex items-center gap-2 text-dashboard-text border-b border-dashboard-border pb-2">
                <Box className="w-5 h-5" />
                <h2 className="font-semibold uppercase tracking-wide text-sm">Entity Stats</h2>
              </div>
              <div className="grid grid-cols-2 gap-3 mt-2">
                <div className="bg-dashboard-bg border border-dashboard-border rounded-lg p-3 flex flex-col items-center">
                  <span className="text-3xl font-mono font-bold text-bmw-cyan">{detections.length}</span>
                  <span className="text-xs uppercase text-dashboard-muted mt-1">Detections</span>
                </div>
                <div className="bg-dashboard-bg border border-dashboard-border rounded-lg p-3 flex flex-col items-center">
                  <span className="text-3xl font-mono font-bold text-dashboard-text">{fps}</span>
                  <span className="text-xs uppercase text-dashboard-muted mt-1">FPS</span>
                </div>
              </div>
              <div className="flex-1 mt-4 border border-dashboard-border bg-dashboard-bg rounded-lg overflow-hidden flex flex-col min-h-[150px]">
                <div className="bg-dashboard-border text-xs uppercase font-semibold p-2 flex justify-between">
                  <span>Event Log</span><span className="text-dashboard-muted">Live</span>
                </div>
                <div className="p-3 text-xs font-mono text-dashboard-muted flex-1 overflow-y-auto flex flex-col gap-1">
                  {events.map((e, i) => (
                    <div key={i} className={e.includes('[ERROR]') ? 'text-bmw-red' : e.includes('[NETWORK]') ? 'text-bmw-cyan' : 'text-dashboard-muted'}>{e}</div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          {/* Dashboard Canvas Area */}
          <div className="col-span-1 lg:col-span-3 hud-container flex flex-col items-center justify-center relative bg-black overflow-hidden group h-full">
            <div className="absolute top-4 left-4 z-10 flex gap-2">
              <span className="bg-dashboard-bg/70 backdrop-blur-sm border border-dashboard-border text-xs px-2 py-1 rounded font-mono uppercase font-semibold flex items-center gap-1">
                <Camera className="w-3 h-3 text-bmw-cyan" /> Camera_01
              </span>
            </div>
            {streamActive ? (
              <div className="w-full h-full relative flex items-center justify-center overflow-hidden">
                <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover" />
                <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
              </div>
            ) : (
              <div className="text-center">
                <Layers className="w-16 h-16 text-dashboard-border mx-auto mb-4" />
                <h2 className="text-xl font-bold text-dashboard-muted uppercase tracking-widest">Stream Offline</h2>
                <p className="text-sm font-mono text-dashboard-border mt-2">Initialize stream from control panel</p>
              </div>
            )}
            {/* Corners */}
            <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-bmw-cyan opacity-50 m-2"></div>
            <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-bmw-cyan opacity-50 m-2"></div>
            <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-bmw-cyan opacity-50 m-2"></div>
            <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-bmw-cyan opacity-50 m-2"></div>
          </div>
        </main>
      ) : (
        <main className="flex-1 flex justify-center items-center">
          <div className="hud-container w-full max-w-3xl flex flex-col gap-6">
            <div className="border-b border-dashboard-border pb-4 flex items-center gap-3">
              <Database className="w-6 h-6 text-bmw-cyan" />
              <h2 className="text-xl font-bold uppercase tracking-wider">Model Training Pipeline</h2>
            </div>

            <div className="grid grid-cols-4 gap-4 items-center mb-4">
              {['Upload Video', 'Extract', 'Annotate', 'Train YOLO'].map((step, i) => {
                const stateMap = ['uploading', 'extracting', 'annotating', 'training'];
                const isActive = stateMap[i] === pipelineState;
                const isDone = pipelineState === 'done' || stateMap.indexOf(pipelineState) > i;
                return (
                  <div key={i} className="flex items-center gap-2">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                      isActive ? 'bg-bmw-cyan text-dashboard-bg animate-pulse' :
                      isDone ? 'bg-green-500 text-dashboard-bg' : 'bg-dashboard-border text-dashboard-muted'
                    }`}>{i + 1}</div>
                    <span className="text-xs uppercase font-semibold text-dashboard-muted">{step}</span>
                  </div>
                );
              })}
            </div>

            <div className="bg-dashboard-bg border border-dashboard-border rounded-lg p-6 border-dashed flex flex-col items-center justify-center min-h-[200px] gap-4">
              <Upload className="w-12 h-12 text-dashboard-muted" />
              <p className="text-dashboard-muted font-mono uppercase text-sm">Select Product Video to Train On</p>
              <div className="flex w-full max-w-sm gap-2">
                <input
                  type="text"
                  placeholder="Product Name (e.g., Yoga Bar Oats)"
                  value={productName}
                  onChange={(e) => setProductName(e.target.value)}
                  className="flex-1 bg-black border border-dashboard-border rounded-md px-4 py-2 text-sm text-dashboard-text focus:outline-none focus:border-bmw-cyan"
                />
                <input
                  type="number"
                  min="5"
                  max="100"
                  value={trainEpochs}
                  onChange={(e) => setTrainEpochs(parseInt(e.target.value) || 10)}
                  className="w-24 bg-black border border-dashboard-border rounded-md px-4 py-2 text-sm text-dashboard-text focus:outline-none focus:border-bmw-cyan"
                  title="Training Epochs"
                />
              </div>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
                className="block w-full text-sm text-center text-dashboard-muted file:bg-dashboard-border file:text-dashboard-text file:border-0 file:py-2 file:px-4 file:rounded-md file:mr-4 file:font-semibold"
              />
            </div>

            <div className="bg-dashboard-border/30 rounded-lg p-4 flex flex-col gap-2 font-mono text-sm min-h-[80px]">
              <span className="text-dashboard-muted uppercase font-bold text-xs">Pipeline Output Log:</span>
              <span className="text-bmw-cyan">{pipelineLog || "Ready."}</span>
            </div>

            <div className="flex justify-end gap-3">
              <button
                disabled={!videoFile || !productName || (pipelineState !== 'idle' && pipelineState !== 'done')}
                className="btn-primary disabled:opacity-50"
                onClick={runPipeline}
              >
                {pipelineState !== 'idle' && pipelineState !== 'done' ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" fill="currentColor" />}
                {pipelineState !== 'idle' && pipelineState !== 'done' ? 'Processing...' : 'Start Pipeline'}
              </button>
            </div>

          </div>
        </main>
      )}
    </div>
  )
}

export default App
