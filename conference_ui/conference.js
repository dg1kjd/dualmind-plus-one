/**
 * PersonaPlex Conference Client
 * Handles WebSocket communication, audio processing, and visualization
 * 
 * Audio architecture:
 * - Browser runs at native sample rate (typically 48kHz) - this is the timing master
 * - opus-recorder handles mic encoding (resamples internally to 24kHz for Opus)
 * - Decoder worker handles server audio (decodes from 24kHz, resamples to browser rate)
 * - No explicit resampling needed in our code - Opus handles it transparently
 */

class ConferenceClient {
    constructor() {
        this.ws = null;
        this.audioContext = null;
        this.recorder = null;
        this.decoderWorker = null;
        this.playbackWorklet = null;
        this.mediaStream = null;
        this.isConnected = false;
        this.isRunning = false;
        
        // Audio levels for visualization
        this.levelA = 0;
        this.levelB = 0;
        this.micLevel = 0;
        
        // Text content
        this.textA = '';
        this.textB = '';
        
        // Visualization
        this.animationFrame = null;
        this.analyser = null;
        this.frequencyData = null;
        this.visualizerDataA = new Float32Array(64);
        this.visualizerDataB = new Float32Array(64);
        
        this.initElements();
        this.initEventListeners();
        this.initVisualizers();
    }
    
    initElements() {
        this.elements = {
            connectionStatus: document.getElementById('connectionStatus'),
            connectionText: document.getElementById('connectionText'),
            micStatus: document.getElementById('micStatus'),
            micIcon: document.getElementById('micIcon'),
            micLevelBar: document.getElementById('micLevelBar'),
            voiceA: document.getElementById('voiceA'),
            voiceB: document.getElementById('voiceB'),
            promptA: document.getElementById('promptA'),
            promptB: document.getElementById('promptB'),
            textA: document.getElementById('textA'),
            textB: document.getElementById('textB'),
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            clearBtn: document.getElementById('clearBtn'),
            visualizerA: document.getElementById('visualizerA'),
            visualizerB: document.getElementById('visualizerB'),
        };
    }
    
    initEventListeners() {
        this.elements.startBtn.addEventListener('click', () => this.start());
        this.elements.stopBtn.addEventListener('click', () => this.stop());
        this.elements.clearBtn.addEventListener('click', () => this.clearText());
        
        // Config changes
        this.elements.voiceA.addEventListener('change', () => this.updateConfig());
        this.elements.voiceB.addEventListener('change', () => this.updateConfig());
    }
    
    initVisualizers() {
        // Setup canvas contexts
        this.ctxA = this.elements.visualizerA.getContext('2d');
        this.ctxB = this.elements.visualizerB.getContext('2d');
        
        // Resize canvases
        const resizeCanvas = (canvas) => {
            canvas.width = canvas.offsetWidth * window.devicePixelRatio;
            canvas.height = canvas.offsetHeight * window.devicePixelRatio;
        };
        resizeCanvas(this.elements.visualizerA);
        resizeCanvas(this.elements.visualizerB);
        
        window.addEventListener('resize', () => {
            resizeCanvas(this.elements.visualizerA);
            resizeCanvas(this.elements.visualizerB);
        });
        
        this.startVisualizerLoop();
    }
    
    startVisualizerLoop() {
        const draw = () => {
            // Update from real spectrum data
            this.updateVisualizersFromSpectrum();
            
            this.drawVisualizer(this.ctxA, this.elements.visualizerA, this.visualizerDataA, '#76b900');
            this.drawVisualizer(this.ctxB, this.elements.visualizerB, this.visualizerDataB, '#00d4ff');
            this.animationFrame = requestAnimationFrame(draw);
        };
        draw();
    }
    
    drawVisualizer(ctx, canvas, data, color) {
        const width = canvas.width;
        const height = canvas.height;
        const barCount = data.length;
        const barWidth = width / barCount;
        const gap = 2;
        
        // Clear
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fillRect(0, 0, width, height);
        
        // Draw bars
        const gradient = ctx.createLinearGradient(0, height, 0, 0);
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, color + '44');
        
        ctx.fillStyle = gradient;
        
        for (let i = 0; i < barCount; i++) {
            const barHeight = data[i] * height;
            const x = i * barWidth + gap / 2;
            const y = height - barHeight;
            
            ctx.beginPath();
            ctx.roundRect(x, y, barWidth - gap, barHeight, 3);
            ctx.fill();
        }
        
        // Add glow effect
        ctx.shadowColor = color;
        ctx.shadowBlur = 15;
        
        // Decay
        for (let i = 0; i < barCount; i++) {
            data[i] *= 0.95;
            if (data[i] < 0.01) data[i] = 0;
        }
    }
    
    updateVisualizersFromSpectrum() {
        if (!this.analyser || !this.frequencyData) return;
        
        // Get real frequency data from analyser
        this.analyser.getByteFrequencyData(this.frequencyData);
        
        // Map frequency bins to visualizer bars (64 bars from 128 bins)
        const binsPerBar = Math.floor(this.frequencyData.length / 64);
        
        for (let i = 0; i < 64; i++) {
            // Average frequency bins for this bar
            let sum = 0;
            for (let j = 0; j < binsPerBar; j++) {
                sum += this.frequencyData[i * binsPerBar + j];
            }
            const normalizedValue = (sum / binsPerBar) / 255;
            
            // Scale by persona levels to create differentiated visualizers
            // A gets more low frequencies, B gets more high frequencies
            const freqBias = i / 64;  // 0 = low freq, 1 = high freq
            const scaleA = this.levelA * (1.2 - freqBias * 0.4);  // Boost lows for A
            const scaleB = this.levelB * (0.8 + freqBias * 0.4);  // Boost highs for B
            
            this.visualizerDataA[i] = Math.max(this.visualizerDataA[i] * 0.85, normalizedValue * scaleA * 8);
            this.visualizerDataB[i] = Math.max(this.visualizerDataB[i] * 0.85, normalizedValue * scaleB * 8);
        }
    }
    
    async start() {
        try {
            this.updateStatus('connecting');
            
            // Initialize audio context at browser's native rate (timing master)
            // No sample rate specified - uses system default (typically 48kHz)
            this.audioContext = new AudioContext();
            await this.audioContext.resume();
            console.log('AudioContext sample rate:', this.audioContext.sampleRate);
            
            // Setup decoder worker for server audio playback
            await this.initDecoder();
            
            // Setup playback worklet with spectrum analyser
            await this.audioContext.audioWorklet.addModule('audio-processor.js');
            this.playbackWorklet = new AudioWorkletNode(this.audioContext, 'conference-processor');
            
            // Create analyser for real spectrum visualization
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;  // 128 frequency bins
            this.analyser.smoothingTimeConstant = 0.7;
            this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
            
            // Connect: worklet -> analyser -> destination
            this.playbackWorklet.connect(this.analyser);
            this.analyser.connect(this.audioContext.destination);
            
            // Decoder worker sends decoded PCM to playback worklet
            // Handle both array format [Float32Array] and direct Float32Array
            this.decodeCount = 0;
            this.decoderWorker.onmessage = (event) => {
                if (!event.data) return;
                
                let samples = null;
                
                // Decoder may return array of channels or direct Float32Array
                if (Array.isArray(event.data)) {
                    // Array format: [[channel0], [channel1], ...] - take first channel
                    samples = event.data[0];
                } else if (event.data instanceof Float32Array) {
                    // Direct Float32Array format
                    samples = event.data;
                } else if (event.data.channelData) {
                    // Object format with channelData property
                    samples = event.data.channelData[0] || event.data.channelData;
                } else if (typeof event.data === 'object' && event.data[0] instanceof Float32Array) {
                    // Array-like object
                    samples = event.data[0];
                }
                
                if (samples && samples.length > 0) {
                    this.decodeCount++;
                    // CRITICAL: Copy samples - decoder worker may reuse its internal buffer
                    const samplesCopy = new Float32Array(samples);
                    
                    if (this.decodeCount <= 5 || this.decodeCount % 100 === 0) {
                        console.log('[DECODE] Decoded samples:', samplesCopy.length, 'count:', this.decodeCount);
                    }
                    
                    this.playbackWorklet.port.postMessage({
                        type: 'audio',
                        samples: samplesCopy
                    });
                }
            };
            
            // Setup opus-recorder for mic encoding
            await this.initRecorder();
            
            // Connect WebSocket
            await this.connectWebSocket();
            
            // Start recording after connection established
            // Recorder.start() will request mic permission and begin encoding
            console.log('Starting recorder...');
            this.recorder.start();
            
            this.elements.startBtn.style.display = 'none';
            this.elements.stopBtn.style.display = 'inline-block';
            this.isRunning = true;
            
        } catch (error) {
            console.error('Failed to start conference:', error);
            this.updateStatus('disconnected');
            alert('Failed to start: ' + error.message);
        }
    }
    
    async initDecoder() {
        return new Promise((resolve, reject) => {
            this.decoderWorker = new Worker('assets/decoderWorker.min.js');
            
            this.decoderWorker.onerror = (e) => {
                console.error('Decoder worker error:', e);
                reject(e);
            };
            
            // Initialize decoder with browser's sample rate
            // Server stream will provide its own BOS page with valid CRC
            this.decoderWorker.postMessage({
                command: 'init',
                bufferLength: Math.round(960 * this.audioContext.sampleRate / 24000),
                decoderSampleRate: 24000,
                outputBufferSampleRate: this.audioContext.sampleRate,
                resampleQuality: 0,  // Fastest
            });
            
            // Allow decoder to initialize, then resolve
            // No warmup BOS needed - server's Ogg stream includes proper headers
            setTimeout(() => {
                console.log('Decoder initialized at', this.audioContext.sampleRate, 'Hz');
                resolve();
            }, 100);
        });
    }
    
    async initRecorder() {
        return new Promise(async (resolve, reject) => {
            try {
                // opus-recorder handles resampling from browser rate to 24kHz internally
                // mediaTrackConstraints tells Recorder to get mic access
                const recorderOptions = {
                    mediaTrackConstraints: {
                        audio: {
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true,
                        }
                    },
                    encoderPath: 'encoderWorker.min.js',
                    bufferLength: Math.round(960 * this.audioContext.sampleRate / 24000),
                    encoderFrameSize: 20,      // 20ms frames
                    encoderSampleRate: 24000,  // Opus encodes at 24kHz
                    maxFramesPerPage: 2,       // Low latency: 2 frames = 40ms
                    numberOfChannels: 1,
                    recordingGain: 1,
                    resampleQuality: 3,
                    encoderComplexity: 0,      // Fastest encoding
                    encoderApplication: 2049,  // VOIP mode
                    streamPages: true,
                };
                
                this.recorder = new Recorder(recorderOptions);
                
                this.recorder.ondataavailable = (opusData) => {
                    // Send Opus data to server
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        const message = new Uint8Array(1 + opusData.length);
                        message[0] = 0x01; // Audio type
                        message.set(new Uint8Array(opusData), 1);
                        this.ws.send(message);
                    }
                };
                
                this.recorder.onstart = () => {
                    console.log('Recorder started');
                    this.elements.micStatus.textContent = 'Mic Active';
                    this.elements.micIcon.classList.add('active');
                };
                
                this.recorder.onstop = () => {
                    console.log('Recorder stopped');
                };
                
                resolve();
            } catch (error) {
                console.error('Recorder init error:', error);
                reject(error);
            }
        });
    }
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host || 'localhost:8999';
            
            const params = new URLSearchParams({
                voice_a: this.elements.voiceA.value,
                voice_b: this.elements.voiceB.value,
                prompt_a: this.elements.promptA.value,
                prompt_b: this.elements.promptB.value,
            });
            
            const url = `${protocol}//${host}/api/conference?${params}`;
            console.log('Connecting to:', url);
            
            this.ws = new WebSocket(url);
            this.ws.binaryType = 'arraybuffer';
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
            };
            
            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket closed');
                this.updateStatus('disconnected');
                this.isConnected = false;
            };
            
            // Wait for handshake
            const checkHandshake = () => {
                if (this.isConnected) {
                    resolve();
                } else {
                    setTimeout(checkHandshake, 100);
                }
            };
            
            // Timeout
            setTimeout(() => {
                if (!this.isConnected) {
                    reject(new Error('Connection timeout'));
                }
            }, 30000);
            
            checkHandshake();
        });
    }
    
    handleMessage(data) {
        const bytes = new Uint8Array(data);
        if (bytes.length === 0) return;
        
        const type = bytes[0];
        const payload = bytes.slice(1);
        
        switch (type) {
            case 0x00: // Handshake
                console.log('Received handshake');
                this.isConnected = true;
                this.updateStatus('connected');
                break;
                
            case 0x01: // Audio
                this.handleAudio(payload);
                break;
                
            case 0x02: // Text
                this.handleText(payload);
                break;
                
            case 0x04: // Levels
                this.handleLevels(payload);
                break;
        }
    }
    
    handleAudio(payload) {
        // Send Opus data to decoder worker
        if (this.decoderWorker) {
            // Check for valid Ogg page (starts with 'OggS')
            const isOgg = payload.length >= 4 && 
                payload[0] === 0x4F && payload[1] === 0x67 && 
                payload[2] === 0x67 && payload[3] === 0x53;
            console.log('[AUDIO] Received opus bytes:', payload.length, isOgg ? '(valid Ogg)' : '(raw)');
            
            this.decoderWorker.postMessage({
                command: 'decode',
                pages: payload,
            });
        }
    }
    
    handleText(payload) {
        // First byte after type indicates which persona (A or B)
        const persona = String.fromCharCode(payload[0]);
        const text = new TextDecoder().decode(payload.slice(1));
        
        if (persona === 'A') {
            this.textA += text;
            this.elements.textA.innerHTML = this.escapeHtml(this.textA) + '<span class="text-cursor"></span>';
            this.elements.textA.scrollTop = this.elements.textA.scrollHeight;
        } else if (persona === 'B') {
            this.textB += text;
            this.elements.textB.innerHTML = this.escapeHtml(this.textB) + '<span class="text-cursor"></span>';
            this.elements.textB.scrollTop = this.elements.textB.scrollHeight;
        }
    }
    
    handleLevels(payload) {
        try {
            const levels = JSON.parse(new TextDecoder().decode(payload));
            this.levelA = levels.level_a || 0;
            this.levelB = levels.level_b || 0;
            // Levels are used by updateVisualizersFromSpectrum() to scale the real spectrum
        } catch (e) {
            console.error('Failed to parse levels:', e);
        }
    }
    
    
    updateConfig() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        
        const config = {
            voice_a: this.elements.voiceA.value,
            voice_b: this.elements.voiceB.value,
            prompt_a: this.elements.promptA.value,
            prompt_b: this.elements.promptB.value,
        };
        
        const payload = new TextEncoder().encode(JSON.stringify(config));
        const message = new Uint8Array(1 + payload.length);
        message[0] = 0x0A; // Config type
        message.set(payload, 1);
        this.ws.send(message);
    }
    
    stop() {
        // Stop recorder
        if (this.recorder) {
            try { this.recorder.stop(); } catch (e) {}
            this.recorder = null;
        }
        
        // Terminate decoder worker
        if (this.decoderWorker) {
            this.decoderWorker.terminate();
            this.decoderWorker = null;
        }
        
        // Close WebSocket
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        // Close audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.playbackWorklet = null;
        this.isRunning = false;
        this.isConnected = false;
        this.updateStatus('disconnected');
        
        this.elements.startBtn.style.display = 'inline-block';
        this.elements.stopBtn.style.display = 'none';
        this.elements.micStatus.textContent = 'Mic Off';
        this.elements.micIcon.classList.remove('active');
        this.elements.micLevelBar.style.width = '0%';
    }
    
    clearText() {
        this.textA = '';
        this.textB = '';
        this.elements.textA.innerHTML = '<span class="text-cursor"></span>';
        this.elements.textB.innerHTML = '<span class="text-cursor"></span>';
    }
    
    updateStatus(status) {
        const statusEl = this.elements.connectionStatus;
        const textEl = this.elements.connectionText;
        
        statusEl.className = 'status-dot';
        
        switch (status) {
            case 'connected':
                statusEl.classList.add('connected');
                textEl.textContent = 'Connected';
                break;
            case 'connecting':
                statusEl.classList.add('connecting');
                textEl.textContent = 'Connecting...';
                break;
            default:
                textEl.textContent = 'Disconnected';
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.conferenceClient = new ConferenceClient();
});
