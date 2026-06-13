import { useState, useEffect, useRef } from 'react'
import { Camera, Layers, Box, Cpu, Play, Square, Activity, Upload, RefreshCw, Database, ShoppingCart, CheckCircle, XCircle, Plus, Minus, Trash2, ScanText, Barcode } from 'lucide-react'
import { BrowserMultiFormatReader, DecodeHintType } from '@zxing/library'
import { io, Socket } from 'socket.io-client'

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'pipeline'>('dashboard');

  // Dashboard Status
  const [isConnected, setIsConnected] = useState(false);
  const [streamActive, setStreamActive] = useState(false);
  const [detections, setDetections] = useState<any[]>([]);
  const [fps, setFps] = useState(0);
  const [events, setEvents] = useState<string[]>(['[SYSTEM] Pipeline ready.', '[SYSTEM] Awaiting stream start.']);

  // Recognition Dialog
  const [recognizedProduct, setRecognizedProduct] = useState<{name: string, confidence: number} | null>(null);
  const dialogCooldownRef = useRef(false);

  // Shopping Cart
  const [cart, setCart] = useState<{name: string, qty: number}[]>([]);
  const [showCartDropdown, setShowCartDropdown] = useState(false);
  const lastHandledRef = useRef<Record<string, number>>({});

  // Barcode / QR Feature
  const [isBarcodeScanning, setIsBarcodeScanning] = useState(false);
  const barcodeScanningRef = useRef(false);
  const [scannedProduct, setScannedProduct] = useState<any>(null);

  // Manual Scan feature
  const scanIdRef = useRef(0);
  const [manualScanState, setManualScanState] = useState<'idle' | 'drawing' | 'identifying'>('idle');
  const [snapshotData, setSnapshotData] = useState<string | null>(null);
  const [manualBox, setManualBox] = useState<{x: number, y: number, w: number, h: number} | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState({x: 0, y: 0});

  const handleManualScanClick = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      setSnapshotData(canvas.toDataURL('image/jpeg', 0.9));
      setManualScanState('drawing');
      setStreamActive(false); 
      setManualBox(null);
    }
  };

  const handleIdentifyManual = async () => {
    if (!snapshotData || !manualBox) return;
    setManualScanState('identifying');
    
    // Scale box coordinates from displayed image size back to original natural image resolution
    const imgElement = document.getElementById('snapshot-image') as HTMLImageElement;
    const scaleX = imgElement.naturalWidth / imgElement.width;
    const scaleY = imgElement.naturalHeight / imgElement.height;
    
    const realBox = {
      x: Math.round(manualBox.x * scaleX),
      y: Math.round(manualBox.y * scaleY),
      w: Math.round(manualBox.w * scaleX),
      h: Math.round(manualBox.h * scaleY)
    };

    const currentScanId = ++scanIdRef.current;

    try {
      // Force a 5-second interval for deep AI analysis as requested
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      // If user cancelled or restarted during the wait, abort update
      if (scanIdRef.current !== currentScanId) return;

      const res = await fetch(`${API_URL}/api/scan_manual`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ image: snapshotData, box: realBox })
      });
      const data = await res.json();
      
      // If user cancelled during fetch, abort update
      if (scanIdRef.current !== currentScanId) return;

      if (data.success && data.class) {
        setRecognizedProduct({ name: data.class, confidence: data.confidence });
        setManualScanState('idle');
        setSnapshotData(null);
        setManualBox(null);
      } else {
        addEvent(`[ERROR] Manual Scan: ${data.message || 'No confident match. Try again.'}`);
        // Redirect user back to box drawing endpoint instead of canceling
        setManualScanState('drawing');
        setManualBox(null);
      }
    } catch (err) {
      if (scanIdRef.current !== currentScanId) return;
      addEvent(`[ERROR] Manual Scan API failed.`);
      setManualScanState('drawing');
      setManualBox(null);
    }
  };

  const cancelManualScan = () => {
    scanIdRef.current++; // Invalidate any running scan promises
    setManualScanState('idle');
    setSnapshotData(null);
    setManualBox(null);
    setStreamActive(true);
  };


  const addToCart = (name: string) => {
    setCart(prev => {
      const existing = prev.find(item => item.name === name);
      if (existing) return prev.map(item => item.name === name ? {...item, qty: item.qty + 1} : item);
      return [...prev, {name, qty: 1}];
    });
  };

  const updateQty = (name: string, delta: number) => {
    setCart(prev => prev.map(item => item.name === name ? {...item, qty: Math.max(1, item.qty + delta)} : item));
  };

  const removeFromCart = (name: string) => {
    setCart(prev => prev.filter(item => item.name !== name));
  };

  const totalItems = cart.reduce((sum, item) => sum + item.qty, 0);

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
      const resUpload = await fetch(`${API_URL}/api/upload`, { method: 'POST', body: formData });
      if (!resUpload.ok) throw new Error("Upload Failed");

      setPipelineLog('Extracting frames...');
      setPipelineState('extracting');
      await fetch(`${API_URL}/api/extract`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoFile.name, target_frames: 180 })
      });

      setPipelineLog('Running auto-annotation...');
      setPipelineState('annotating');
      await fetch(`${API_URL}/api/annotate/auto`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoFile.name, class_id: 0 })
      });

      setPipelineLog(`Training YOLOv8 (${trainEpochs} Epochs)...`);
      setPipelineState('training');
      const resTrain = await fetch(`${API_URL}/api/train`, {
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

  const API_URL = `${window.location.protocol}//${window.location.hostname}:5000`;

  useEffect(() => {
    socketRef.current = io(API_URL);
    socketRef.current.on('connect', () => { setIsConnected(true); addEvent('[NETWORK] Connected to Inference Engine API.'); });
    socketRef.current.on('disconnect', () => { setIsConnected(false); addEvent('[NETWORK] Disconnected from API.'); });
    socketRef.current.on('status', (data) => addEvent(`[SYSTEM] ${data.data}`));
    socketRef.current.on('detections', (data) => {
      framesRef.current += 1;
      const now = performance.now();
      if (now - lastTimeRef.current >= 1000) {
        setFps(Math.round((framesRef.current * 1000) / (now - lastTimeRef.current)));
        framesRef.current = 0;
        lastTimeRef.current = now;
      }
      if (data.error) {
        if (!data.error.includes('No model loaded')) addEvent(`[ERROR] ${data.error}`);
        setDetections([]);
        return;
      }
      setDetections(data.detections || []);
    });
    return () => { socketRef.current?.disconnect(); };
  }, []);



  useEffect(() => {
    if (activeTab !== 'dashboard' || !streamActive || dialogCooldownRef.current || recognizedProduct || detections.length === 0) return;

    // Filter out items already in the cart OR on the 30s cooldown
    const nowTime = Date.now();
    const candidateDetections = detections.filter(d => {
      const isInCart = cart.some(item => item.name === d.class);
      if (isInCart) return false;

      const lastTime = lastHandledRef.current[d.class] || 0;
      return (nowTime - lastTime > 30000);
    });

    if (candidateDetections.length === 0) return;

    // Pick the best from the REMAINING candidates
    const best = candidateDetections.reduce((a: any, b: any) => a.confidence > b.confidence ? a : b);
    
    // Lower threshold to 30% for better sensitivity to objects like bottles/bowls
    if (best.confidence >= 0.3) {
      dialogCooldownRef.current = true;
      setRecognizedProduct({ name: best.class, confidence: best.confidence });
    }
  }, [detections, cart, activeTab, streamActive, recognizedProduct]);

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
            height: { ideal: 1080 },
            facingMode: 'environment'
          }
        });
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        addEvent('[HARDWARE] Camera initialized.');

        const hints = new Map();
        hints.set(DecodeHintType.TRY_HARDER, true);
        const zxingReader = new BrowserMultiFormatReader(hints);
        let nativeBarcodeDetector: any = null;
        if ('BarcodeDetector' in window) {
           nativeBarcodeDetector = new (window as any).BarcodeDetector();
        }
        
        const sendFrame = () => {
          if (!streamRef.current || !videoRef.current || !socketRef.current) return;

          if (barcodeScanningRef.current) {
             setDetections([]);
             
             const processCode = (code: string) => {
                 barcodeScanningRef.current = false;
                 setIsBarcodeScanning(false);
                 addEvent(`[SYSTEM] Scanned code: ${code}. Fetching data...`);
                 
                 fetch(`https://in.openfoodfacts.org/api/v0/product/${code}.json`)
                   .then(r => r.json())
                   .then(data => {
                       if (data.status === 1 && data.product) {
                           const p = data.product;
                           setScannedProduct({
                               id: code,
                               name: p.product_name || p.generic_name || 'Unknown Product',
                               brand: p.brands ? p.brands.split(',')[0] : 'Unknown Brand',
                               image: p.image_url || p.image_front_url,
                               isVeg: p.ingredients_analysis_tags ? 
                                      (p.ingredients_analysis_tags.includes('en:vegetarian') ? true : 
                                      (p.ingredients_analysis_tags.includes('en:non-vegetarian') ? false : null)) : null,
                               nutriScore: p.nutriscore_grade ? p.nutriscore_grade.toUpperCase() : 'N/A'
                           });
                       } else {
                          addEvent(`[ERROR] Product ${code} not found in database.`);
                          barcodeScanningRef.current = true;
                          setIsBarcodeScanning(true);
                       }
                   }).catch(() => {
                       addEvent(`[ERROR] Network failed reaching OpenFoodFacts.`);
                       barcodeScanningRef.current = true;
                       setIsBarcodeScanning(true);
                   });
                 
                 // CRITICAL FIX: Keep the background loop alive after a successful scan!
                 requestRef.current = setTimeout(() => { if (streamRef.current) sendFrame(); }, 100);
             };

             const runZxing = () => {
                 const hiddenCanvas = document.createElement('canvas');
                 const vw = videoRef.current!.videoWidth || 854;
                 const vh = videoRef.current!.videoHeight || 480;
                 
                 hiddenCanvas.width = vw;
                 hiddenCanvas.height = vh;
                 
                 const ctx = hiddenCanvas.getContext('2d');
                 if (ctx) {
                     ctx.imageSmoothingEnabled = false;
                     
                     // Draw the full video frame without cropping
                     ctx.drawImage(videoRef.current!, 0, 0, vw, vh);
                     
                     const img = new Image();
                     img.src = hiddenCanvas.toDataURL('image/png');
                     img.onload = async () => {
                         if (!barcodeScanningRef.current) return;
                         try {
                             const result = await zxingReader.decodeFromImageElement(img);
                             if (result) {
                                 processCode(result.getText());
                             } else {
                                 // Add a heartbeat log every roughly 30 frames so we know it's actively scanning
                                 if (Math.random() < 0.03) addEvent('[SYSTEM] Scanner analyzing frame (move camera to focus)...');
                                 requestRef.current = setTimeout(() => { if (streamRef.current) sendFrame(); }, 300);
                             }
                         } catch (e) {
                             if (Math.random() < 0.03) addEvent('[SYSTEM] Scanner analyzing frame (move camera to focus)...');
                             requestRef.current = setTimeout(() => { if (streamRef.current) sendFrame(); }, 300);
                         }
                     };
                     img.onerror = () => {
                         requestRef.current = setTimeout(() => { if (streamRef.current) sendFrame(); }, 300);
                     };
                 } else {
                     requestRef.current = setTimeout(() => { if (streamRef.current) sendFrame(); }, 300);
                 }
             };

             if (nativeBarcodeDetector) {
                 // Native BarcodeDetector works massively better on raw, full-res DOM elements
                 nativeBarcodeDetector.detect(videoRef.current)
                     .then((barcodes: any[]) => {
                         if (barcodes.length > 0 && barcodeScanningRef.current) {
                             processCode(barcodes[0].rawValue);
                         } else {
                             runZxing();
                         }
                     })
                     .catch((e: any) => {
                         console.error("BarcodeDetector Error:", e);
                         runZxing();
                     });
             } else {
                 runZxing();
             }
             return;
          }

          const hiddenCanvas = document.createElement('canvas');
          
          // Calculate scale to keep the max resolution around 480p equivalent without crushing the aspect ratio!
          const scale = Math.min(854 / videoRef.current.videoWidth, 480 / videoRef.current.videoHeight);
          const targetWidth = Math.round(videoRef.current.videoWidth * scale);
          const targetHeight = Math.round(videoRef.current.videoHeight * scale);

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
      <header className="flex justify-between items-center hud-container py-3 relative z-50">
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

        <div className="flex gap-4 items-center relative">
          {/* Cart Badge & Dropdown */}
          <div className="relative">
            <button 
              onClick={() => setShowCartDropdown(!showCartDropdown)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all ${showCartDropdown ? 'bg-bmw-cyan text-dashboard-bg border-bmw-cyan' : 'bg-dashboard-border text-dashboard-text border-bmw-cyan/30 hover:border-bmw-cyan'}`}
            >
              <ShoppingCart className="w-4 h-4" />
              <span className="text-sm font-bold">{totalItems} item{totalItems !== 1 ? 's' : ''}</span>
              {totalItems > 0 && (
                <span className={`absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full text-[10px] font-bold flex items-center justify-center ${showCartDropdown ? 'bg-white text-bmw-cyan' : 'bg-bmw-cyan text-dashboard-bg'}`}>
                  {cart.length}
                </span>
              )}
            </button>

            {/* Dropdown Menu */}
            {showCartDropdown && (
              <div className="absolute top-full right-0 mt-2 w-72 bg-dashboard-bg border border-dashboard-border rounded-xl shadow-2xl z-[100] overflow-hidden" style={{boxShadow: '0 10px 30px rgba(0,0,0,0.5)'}}>
                <div className="bg-dashboard-border p-3 flex justify-between items-center">
                  <span className="text-xs font-bold uppercase tracking-wider text-dashboard-text">Shopping Cart</span>
                  <button onClick={() => setShowCartDropdown(false)} className="text-dashboard-muted hover:text-white"><XCircle className="w-4 h-4" /></button>
                </div>
                <div className="max-h-96 overflow-y-auto p-2 flex flex-col gap-2">
                  {cart.length === 0 ? (
                    <div className="py-8 text-center text-dashboard-muted text-sm italic">Your cart is empty</div>
                  ) : (
                    cart.map(item => (
                      <div key={item.name} className="bg-dashboard-border/30 rounded-lg p-3 flex flex-col gap-2 border border-transparent hover:border-bmw-cyan/20 transition-all">
                        <div className="flex justify-between items-start">
                          <span className="text-sm font-semibold capitalize text-dashboard-text truncate">{item.name.replace(/_/g, ' ')}</span>
                          <button onClick={() => removeFromCart(item.name)} className="text-dashboard-muted hover:text-bmw-red transition-colors"><Trash2 className="w-3.5 h-3.5" /></button>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-dashboard-muted font-mono uppercase">Quantity</span>
                          <div className="flex items-center gap-3">
                            <button onClick={() => updateQty(item.name, -1)} className="w-6 h-6 rounded bg-dashboard-border flex items-center justify-center hover:bg-bmw-cyan/20 text-dashboard-text transition-colors"><Minus className="w-3 h-3" /></button>
                            <span className="text-sm font-bold text-bmw-cyan min-w-[20px] text-center">{item.qty}</span>
                            <button onClick={() => updateQty(item.name, 1)} className="w-6 h-6 rounded bg-dashboard-border flex items-center justify-center hover:bg-bmw-cyan/20 text-dashboard-text transition-colors"><Plus className="w-3 h-3" /></button>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
                {cart.length > 0 && (
                  <div className="p-3 bg-dashboard-bg border-t border-dashboard-border">
                    <button className="w-full py-2 bg-bmw-cyan text-dashboard-bg font-bold uppercase text-xs tracking-widest rounded-lg hover:bg-cyan-400 transition-all">
                      Checkout Total: {totalItems} Units
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

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

      {/* Barcode Product Dialog Modal */}
      {scannedProduct && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-dashboard-bg border border-bmw-cyan/50 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-w-sm w-full mx-4" style={{boxShadow: '0 0 40px rgba(0,204,255,0.2)'}}>
            {scannedProduct.image ? (
              <div className="h-48 w-full bg-white flex items-center justify-center p-4 relative">
                <img src={scannedProduct.image} alt={scannedProduct.name} className="h-full object-contain mix-blend-multiply" />
                {scannedProduct.isVeg !== null && (
                  <div className="absolute top-4 right-4 bg-white rounded-md p-1 shadow-md border border-gray-200">
                    <div className={`w-5 h-5 border-[1.5px] flex items-center justify-center ${scannedProduct.isVeg ? 'border-green-600' : 'border-red-600'}`}>
                      <div className={`w-2.5 h-2.5 rounded-full ${scannedProduct.isVeg ? 'bg-green-600' : 'bg-red-600'}`}></div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-32 w-full bg-dashboard-border flex items-center justify-center text-dashboard-muted">
                No Image Available
              </div>
            )}
            <div className="p-6 flex flex-col gap-4">
              <div className="text-center">
                <p className="text-xs uppercase tracking-widest text-dashboard-muted font-mono mb-1">{scannedProduct.brand}</p>
                <h2 className="text-xl font-bold text-dashboard-text capitalize leading-tight">{scannedProduct.name}</h2>
                <div className="flex items-center justify-center gap-2 mt-2">
                  <span className="text-xs px-2 py-0.5 rounded bg-dashboard-border text-dashboard-muted font-mono">ID: {scannedProduct.id}</span>
                  {scannedProduct.nutriScore !== 'N/A' && (
                    <span className="text-xs px-2 py-0.5 rounded bg-bmw-cyan/20 text-bmw-cyan font-bold">NutriScore: {scannedProduct.nutriScore}</span>
                  )}
                </div>
              </div>
              <div className="flex gap-3 w-full mt-2">
                <button
                  className="flex-1 py-3 rounded-xl bg-bmw-cyan text-dashboard-bg font-bold uppercase text-xs tracking-wider flex items-center justify-center gap-2 hover:bg-cyan-400 transition-all"
                  onClick={() => {
                    addToCart(scannedProduct.name);
                    setScannedProduct(null);
                  }}
                >
                  <CheckCircle className="w-4 h-4" /> Add to Cart
                </button>
                <button
                  className="px-4 py-3 rounded-xl bg-dashboard-border text-dashboard-muted font-bold uppercase flex items-center justify-center hover:text-dashboard-text transition-all"
                  onClick={() => setScannedProduct(null)}
                >
                  <XCircle className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recognition Dialog Modal */}
      {recognizedProduct && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-dashboard-bg border border-bmw-cyan/50 rounded-2xl shadow-2xl p-8 flex flex-col items-center gap-6 max-w-sm w-full mx-4" style={{boxShadow: '0 0 40px rgba(0,204,255,0.2)'}}>
            <div className="w-16 h-16 rounded-full bg-bmw-cyan/10 border-2 border-bmw-cyan flex items-center justify-center">
              <CheckCircle className="w-8 h-8 text-bmw-cyan" />
            </div>
            <div className="text-center">
              <p className="text-xs uppercase tracking-widest text-dashboard-muted font-mono mb-1">Product Recognized</p>
              <h2 className="text-2xl font-bold text-dashboard-text capitalize">{recognizedProduct.name.replace(/_/g, ' ')}</h2>
              <p className="text-sm text-bmw-cyan font-mono mt-1">{(recognizedProduct.confidence * 100).toFixed(1)}% confidence</p>
            </div>
            <div className="flex gap-3 w-full">
              <button
                className="flex-1 py-3 rounded-xl bg-bmw-cyan text-dashboard-bg font-bold uppercase tracking-wider flex items-center justify-center gap-2 hover:bg-cyan-400 transition-all"
                onClick={() => {
                  addToCart(recognizedProduct.name);
                  lastHandledRef.current[recognizedProduct.name] = Date.now();
                  setRecognizedProduct(null);
                  setTimeout(() => { dialogCooldownRef.current = false; }, 3000);
                }}
              >
                <CheckCircle className="w-4 h-4" /> Yes, Add to Cart
              </button>
              <button
                className="flex-1 py-3 rounded-xl bg-dashboard-border text-dashboard-muted font-bold uppercase tracking-wider flex items-center justify-center gap-2 hover:text-dashboard-text transition-all"
                onClick={() => {
                  lastHandledRef.current[recognizedProduct.name] = Date.now();
                  setRecognizedProduct(null);
                  setTimeout(() => { dialogCooldownRef.current = false; }, 2000);
                }}
              >
                <XCircle className="w-4 h-4" /> Try Again
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      {activeTab === 'dashboard' ? (
        <main className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 min-h-0 overflow-hidden">
          {/* Dashboard Left Panel */}
          <div className="col-span-1 flex flex-col min-h-0">
            <div className="hud-container-accent flex-1 flex flex-col gap-4 overflow-y-auto pr-1 custom-scrollbar">
              <div className="flex items-center gap-2 text-bmw-cyan border-b border-dashboard-border pb-2 shrink-0">
                <Activity className="w-5 h-5" />
                <h2 className="font-semibold uppercase tracking-wide text-sm">Controls</h2>
              </div>
              <div className="flex flex-col gap-3 mt-2 shrink-0">
                <button
                  className={`py-3 rounded-lg font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-all shadow-lg ${streamActive ? 'bg-bmw-red hover:bg-red-600 text-white' : 'bg-bmw-cyan hover:bg-cyan-500 text-dashboard-bg'}`}
                  onClick={() => setStreamActive(!streamActive)}
                >
                  {streamActive ? <Square className="w-5 h-5" /> : <Play className="w-5 h-5" fill="currentColor" />}
                  {streamActive ? 'Stop Stream' : 'Start Stream'}
                </button>
                <button
                  className={`py-2 rounded-lg font-bold uppercase text-xs tracking-wider border flex items-center justify-center gap-2 transition-all ${!streamActive ? 'opacity-50 cursor-not-allowed border-dashboard-border text-dashboard-muted' : isBarcodeScanning ? 'bg-bmw-cyan text-dashboard-bg border-bmw-cyan shadow-[0_0_15px_rgba(0,204,255,0.5)]' : 'bg-dashboard-bg border-bmw-cyan text-bmw-cyan hover:bg-bmw-cyan/10'}`}
                  disabled={!streamActive}
                  onClick={() => {
                    const next = !isBarcodeScanning;
                    setIsBarcodeScanning(next);
                    barcodeScanningRef.current = next;
                    if (next) addEvent('[SYSTEM] Barcode scanner engaged. Hold product to camera.');
                  }}
                >
                  <Barcode className="w-4 h-4" />
                  {isBarcodeScanning ? 'Scanning Barcode...' : 'Scan Barcode / QR'}
                </button>
                <button
                  className={`btn-secondary uppercase text-xs tracking-wider ${activeModel === 'default' ? 'bg-dashboard-border text-white border-bmw-cyan border' : ''} ${modelLoading !== null ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={modelLoading !== null}
                  onClick={() => {
                    setModelLoading('default');
                    fetch(`${API_URL}/api/model/load`, {
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
                    fetch(`${API_URL}/api/model/load`, {
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
              <div className="mt-6 flex items-center gap-2 text-dashboard-text border-b border-dashboard-border pb-2 shrink-0">
                <Box className="w-5 h-5" />
                <h2 className="font-semibold uppercase tracking-wide text-sm">Entity Stats</h2>
              </div>
              <div className="grid grid-cols-2 gap-3 mt-2 shrink-0">
                <div className="bg-dashboard-bg border border-dashboard-border rounded-lg p-3 flex flex-col items-center">
                  <span className="text-3xl font-mono font-bold text-bmw-cyan">{detections.length}</span>
                  <span className="text-xs uppercase text-dashboard-muted mt-1">Detections</span>
                </div>
                <div className="bg-dashboard-bg border border-dashboard-border rounded-lg p-3 flex flex-col items-center">
                  <span className="text-3xl font-mono font-bold text-dashboard-text">{fps}</span>
                  <span className="text-xs uppercase text-dashboard-muted mt-1">FPS</span>
                </div>
              </div>
              <div className="flex-1 mt-4 border border-dashboard-border bg-dashboard-bg rounded-lg overflow-hidden flex flex-col min-h-[250px] shrink-0">
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
            {streamActive && manualScanState === 'idle' && (
              <button 
                className="absolute top-4 right-4 z-10 bg-dashboard-bg/80 hover:bg-bmw-cyan text-dashboard-text hover:text-dashboard-bg border border-dashboard-border backdrop-blur-sm text-xs px-3 py-1.5 rounded-lg font-bold uppercase transition-colors flex items-center gap-2"
                onClick={handleManualScanClick}
              >
                <ScanText className="w-4 h-4" /> Scan Manually
              </button>
            )}
            {isBarcodeScanning && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center pointer-events-none">
                <div className="w-64 h-64 border-4 border-bmw-cyan/50 rounded-2xl relative shadow-[0_0_50px_rgba(0,204,255,0.2)]">
                  <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-bmw-cyan rounded-tl-xl -m-[4px]"></div>
                  <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-bmw-cyan rounded-tr-xl -m-[4px]"></div>
                  <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-bmw-cyan rounded-bl-xl -m-[4px]"></div>
                  <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-bmw-cyan rounded-br-xl -m-[4px]"></div>
                  <div className="w-full h-1 bg-bmw-cyan animate-[bounce_2s_infinite] shadow-[0_0_10px_#00ccff]"></div>
                </div>
                <div className="mt-6 bg-black/50 backdrop-blur-md px-4 py-2 rounded-full border border-dashboard-border">
                  <p className="text-white font-mono text-sm uppercase tracking-widest animate-[pulse_1s_infinite]">Scan Code To Add Item</p>
                </div>
              </div>
            )}
            
            {manualScanState !== 'idle' && snapshotData ? (
              <div className="w-full h-full relative flex flex-col items-center justify-center p-4">
                <h3 className="text-dashboard-text font-bold uppercase mb-2">Draw Box Around Product</h3>
                <div 
                  className="relative cursor-crosshair inline-block max-w-full max-h-[80%]"
                  onMouseDown={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    setIsDrawing(true);
                    setDrawStart({x, y});
                    setManualBox({x, y, w: 0, h: 0});
                  }}
                  onMouseMove={(e) => {
                    if (!isDrawing) return;
                    const rect = e.currentTarget.getBoundingClientRect();
                    const currentX = e.clientX - rect.left;
                    const currentY = e.clientY - rect.top;
                    setManualBox({
                      x: Math.min(drawStart.x, currentX),
                      y: Math.min(drawStart.y, currentY),
                      w: Math.abs(currentX - drawStart.x),
                      h: Math.abs(currentY - drawStart.y)
                    });
                  }}
                  onMouseUp={() => setIsDrawing(false)}
                  onMouseLeave={() => setIsDrawing(false)}
                >
                  <img id="snapshot-image" src={snapshotData} className="max-w-full max-h-full object-contain pointer-events-none" alt="Snapshot" />
                  {manualBox && (
                    <div 
                      className="absolute border-2 border-bmw-cyan bg-bmw-cyan/20 pointer-events-none"
                      style={{
                        left: manualBox.x,
                        top: manualBox.y,
                        width: manualBox.w,
                        height: manualBox.h
                      }}
                    />
                  )}
                </div>
                <div className="flex gap-4 mt-4">
                  <button onClick={cancelManualScan} className="btn-secondary">Cancel</button>
                  <button 
                    onClick={handleIdentifyManual} 
                    className="btn-primary" 
                    disabled={manualScanState === 'identifying' || !manualBox || manualBox.w < 10 || manualBox.h < 10}
                  >
                    {manualScanState === 'identifying' ? 'Identifying...' : 'Identify Product'}
                  </button>
                </div>
              </div>
            ) : streamActive ? (
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
