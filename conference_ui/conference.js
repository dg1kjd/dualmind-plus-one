// Title: DualMind+1 Moshi Edition Conference Interface
// Description: This script manages the real-time audio conferencing interface for DualMind+1 Moshi Edition, handling user interactions and audio stream coordination.
// Author: Jens David
// Copyright: 2026 Jens David Consulting
// License: MIT

/**
 * DualMind - MoshiPlex Edition
 * Real-time dual AI conversation interface
 * 
 * Audio architecture:
 * - Browser runs at native sample rate (typically 48kHz) - this is the timing master
 * - opus-recorder handles mic encoding (resamples internally to 24kHz for Opus)
 * - Decoder worker handles server audio (decodes from 24kHz, resamples to browser rate)
 * - No explicit resampling needed in our code - Opus handles it transparently
 */

// Personality presets
const PRESETS = {
    philosopher: "You are a deep thinker who loves exploring existential questions. You often quote philosophers and challenge assumptions. You find meaning in paradoxes and enjoy Socratic dialogue.",
    comedian: "You are a witty comedian who finds humor in everything. You love puns, wordplay, and absurdist observations. You keep the mood light but can be surprisingly insightful.",
    detective: "You are a grizzled police detective investigating a candy theft. A little girl reported her lollipop stolen. You have a prime suspect and you're going to get to the bottom of this. You take this case VERY seriously, using intense interrogation tactics for this heinous crime.",
    robber: "You are a sneaky but not very bright candy thief. You definitely stole that little girl's lollipop but you're trying to act innocent. You're bad at lying - you keep accidentally revealing incriminating details. You get defensive and make increasingly absurd excuses.",
    poet: "You are a romantic soul who sees beauty everywhere. You speak in metaphors and occasionally break into verse. You're deeply emotional and find poetry in mundane moments.",
    mlengineer: "You are an ML engineer obsessed with transformers and gradient descent. You relate everything to neural networks and loss functions. You complain about GPU memory, CUDA errors, and training instability. You casually drop terms like 'attention heads' and 'batch normalization' into conversation.",
    psychologist: "You are a practicing psychologist who can't stop analyzing everyone. You ask probing questions about childhood and feelings. You say 'and how does that make you feel?' a lot. You take notes and occasionally say 'mmhmm, interesting' while nodding thoughtfully."
};

// Determine base path dynamically so reverse-proxied deployments under /suburl/ work.
const BASE_PATH = (() => {
    const { pathname } = window.location;
    if (pathname.endsWith('/')) {
        return pathname;
    }
    const segments = pathname.split('/');
    segments.pop();
    let base = segments.join('/') || '/';
    if (!base.endsWith('/')) {
        base += '/';
    }
    return base;
})();

const resolvePath = (relativePath) => {
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(relativePath)) {
        return relativePath; // Already absolute (http:, https:, etc.)
    }
    if (relativePath.startsWith('/')) {
        return relativePath; // Absolute on current origin
    }
    return BASE_PATH + relativePath;
};

const DISABLED_VOICE_VALUE = '__disabled__';

// Global preset setter
function setPreset(persona, presetName) {
    const voiceSelect = document.getElementById(`voice${persona}`);
    const textarea = document.getElementById(`prompt${persona}`);
    if (voiceSelect && voiceSelect.value === DISABLED_VOICE_VALUE) {
        return;
    }
    if (textarea && textarea.disabled) {
        return;
    }
    const preset = PRESETS[presetName];
    if (preset) {
        const textarea = document.getElementById(`prompt${persona}`);
        if (textarea) {
            textarea.value = preset;
            if (window.conferenceClient) {
                window.conferenceClient.updateConfig();
            }
        }
    }
}

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
        this.micAnalyser = null;
        this.frequencyData = null;
        this.micFrequencyData = null;
        this.visualizerDataA = new Float32Array(64);
        this.visualizerDataB = new Float32Array(64);
        this.visualizerDataMic = new Float32Array(64);
        this.personaPanels = {
            A: document.querySelector('[data-persona-panel="A"]'),
            B: document.querySelector('[data-persona-panel="B"]'),
        };
        this.presetContainers = {
            A: document.querySelector('[data-persona-buttons="A"]'),
            B: document.querySelector('[data-persona-buttons="B"]'),
        };

        // Stats tracking
        this.stats = {
            startTime: null,
            frames: 0,
            dropped: 0,
            lastPingTime: 0,
            latency: 0
        };
        this.statsInterval = null;
        
        // Recording buffers (Float32Arrays accumulated during session)
        this.recordingBuffers = {
            mixedAudio: [],   // Combined A+B for playback channel
            micAudio: []      // Mic input
        };
        this.recordingSampleRate = 48000; // Will be set from audioContext
        this.hasRecording = false;
        this.miclessMode = false;
        
        this.initElements();
        this.initEventListeners();
        this.initVisualizers();
        this.refreshPersonaState('A');
        this.refreshPersonaState('B');
    }
    
    initElements() {
        this.elements = {
            connectionStatus: document.getElementById('connectionStatus'),
            connectionText: document.getElementById('connectionText'),
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
            visualizerMic: document.getElementById('visualizerMic'),
            // Stats elements
            statTime: document.getElementById('statTime'),
            statLevelA: document.getElementById('statLevelA'),
            statLevelB: document.getElementById('statLevelB'),
            statLatency: document.getElementById('statLatency'),
            statFrames: document.getElementById('statFrames'),
            statDropped: document.getElementById('statDropped'),
            downloadBtn: document.getElementById('downloadBtn'),
        };
    }
    
    initEventListeners() {
        this.elements.startBtn.addEventListener('click', () => this.start());
        this.elements.stopBtn.addEventListener('click', () => this.stop());
        this.elements.clearBtn.addEventListener('click', () => this.clearText());
        this.elements.downloadBtn.addEventListener('click', () => this.downloadRecording());
        
        // Config changes
        this.elements.voiceA.addEventListener('change', () => {
            this.refreshPersonaState('A');
            this.updateConfig();
        });
        this.elements.voiceB.addEventListener('change', () => {
            this.refreshPersonaState('B');
            this.updateConfig();
        });
    }
    
    initVisualizers() {
        // Setup canvas contexts
        this.ctxA = this.elements.visualizerA.getContext('2d');
        this.ctxB = this.elements.visualizerB.getContext('2d');
        this.ctxMic = this.elements.visualizerMic.getContext('2d');
        
        // Resize canvases
        const resizeCanvas = (canvas) => {
            if (canvas) {
                canvas.width = canvas.offsetWidth * window.devicePixelRatio;
                canvas.height = canvas.offsetHeight * window.devicePixelRatio;
            }
        };
        resizeCanvas(this.elements.visualizerA);
        resizeCanvas(this.elements.visualizerB);
        resizeCanvas(this.elements.visualizerMic);
        
        window.addEventListener('resize', () => {
            resizeCanvas(this.elements.visualizerA);
            resizeCanvas(this.elements.visualizerB);
            resizeCanvas(this.elements.visualizerMic);
        });
        
        this.startVisualizerLoop();
    }
    
    startVisualizerLoop() {
        const draw = () => {
            // Update from real spectrum data
            this.updateVisualizersFromSpectrum();
            this.updateMicVisualizer();
            
            this.drawVisualizer(this.ctxA, this.elements.visualizerA, this.visualizerDataA, '#76b900');
            this.drawVisualizer(this.ctxB, this.elements.visualizerB, this.visualizerDataB, '#00d4ff');
            this.drawVisualizer(this.ctxMic, this.elements.visualizerMic, this.visualizerDataMic, '#ff9500');
            this.animationFrame = requestAnimationFrame(draw);
        };
        draw();
    }
    
    updateMicVisualizer() {
        if (!this.micAnalyser || !this.micFrequencyData) return;
        
        this.micAnalyser.getByteFrequencyData(this.micFrequencyData);
        
        const binsPerBar = Math.floor(this.micFrequencyData.length / 64);
        
        for (let i = 0; i < 64; i++) {
            let sum = 0;
            for (let j = 0; j < binsPerBar; j++) {
                sum += this.micFrequencyData[i * binsPerBar + j];
            }
            const normalizedValue = (sum / binsPerBar) / 255;
            this.visualizerDataMic[i] = Math.max(this.visualizerDataMic[i] * 0.85, normalizedValue);
        }
    }
    
    updateStats() {
        if (!this.stats.startTime) return;
        
        // Session time
        const elapsed = Math.floor((Date.now() - this.stats.startTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = (elapsed % 60).toString().padStart(2, '0');
        this.elements.statTime.textContent = `${mins}:${secs}`;
        
        // Levels in dBFS (20 * log10(level), clamped to reasonable range)
        this.elements.statLevelA.textContent = this.levelToDbfs(this.levelA);
        this.elements.statLevelB.textContent = this.levelToDbfs(this.levelB);
        
        // Latency
        this.elements.statLatency.textContent = this.stats.latency > 0 ? this.stats.latency : '--';
        
        // Frames
        this.elements.statFrames.textContent = this.stats.frames;
        this.elements.statDropped.textContent = this.stats.dropped;
    }
    
    levelToDbfs(level) {
        if (level <= 0.0001) return '--- dBFS';
        const dbfs = 20 * Math.log10(level);
        // Clamp to reasonable display range
        const clamped = Math.max(-60, Math.min(0, dbfs));
        return `${clamped.toFixed(0)} dBFS`;
    }
    
    resetStatsDisplay() {
        this.elements.statTime.textContent = '00:00';
        this.elements.statLevelA.textContent = '--- dBFS';
        this.elements.statLevelB.textContent = '--- dBFS';
        this.elements.statLatency.textContent = '--';
        this.elements.statFrames.textContent = '0';
        this.elements.statDropped.textContent = '0';
    }
    
    drawVisualizer(ctx, canvas, data, color) {
        if (!ctx || !canvas) return;
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
        const disabledA = this.isPersonaDisabled('A');
        const disabledB = this.isPersonaDisabled('B');
        if (disabledA && disabledB) {
            alert('Please enable at least one AI mind before starting.');
            return;
        }
        try {
            this.updateStatus('connecting');
            
            // Initialize audio context at browser's native rate (timing master)
            // No sample rate specified - uses system default (typically 48kHz)
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) {
                throw new Error('AudioContext is not supported in this browser.');
            }
            this.audioContext = new AudioCtx();
            await this.audioContext.resume();
            console.log('AudioContext sample rate:', this.audioContext.sampleRate);
            
            // Setup decoder worker for server audio playback
            await this.initDecoder();
            
            // Setup playback worklet with spectrum analyser
            await this.audioContext.audioWorklet.addModule(resolvePath('audio-processor.js'));
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
                    
                    // Record the mixed audio (A+B combined from server)
                    this.recordingBuffers.mixedAudio.push(new Float32Array(samplesCopy));
                    
                    this.playbackWorklet.port.postMessage({
                        type: 'audio',
                        samples: samplesCopy
                    });
                }
            };
            
            // Setup opus-recorder for mic encoding (falls back to mic-less mode)
            await this.initRecorder();
            
            // Connect WebSocket
            await this.connectWebSocket();
            
            if (!this.miclessMode && this.recorder) {
                // Start recording after connection established
                // Recorder.start() will request mic permission and begin encoding
                console.log('Starting recorder...');
                this.recorder.start();
                
                // Setup mic analyzer after recorder starts (needs the media stream)
                await this.setupMicAnalyser();
            } else {
                console.warn('Microphone unavailable; running AI-only mode.');
            }
            
            // Start stats tracking
            this.stats.startTime = Date.now();
            this.stats.frames = 0;
            this.stats.dropped = 0;
            this.statsInterval = setInterval(() => this.updateStats(), 100);
            
            // Clear recording buffers for new session
            this.recordingBuffers = { mixedAudio: [], micAudio: [] };
            this.recordingSampleRate = this.audioContext.sampleRate; // Use actual browser sample rate
            this.hasRecording = false;
            this.elements.downloadBtn.disabled = true;
            
            this.elements.startBtn.style.display = 'none';
            this.elements.stopBtn.style.display = 'inline-block';
            this.isRunning = true;
            this.setVoiceSelectorsEnabled(false);
            this.refreshPersonaState('A');
            this.refreshPersonaState('B');
            
        } catch (error) {
            console.error('Failed to start conference:', error);
            this.updateStatus('disconnected');
            alert('Failed to start: ' + error.message);
        }
    }
    
    async setupMicAnalyser() {
        // Setup analyser for mic spectrum visualization
        if (this.mediaStream) {
            this.micAnalyser = this.audioContext.createAnalyser();
            this.micAnalyser.fftSize = 256;
            this.micAnalyser.smoothingTimeConstant = 0.7;
            this.micFrequencyData = new Uint8Array(this.micAnalyser.frequencyBinCount);
            
            // Connect mic source to analyser
            const analyserSource = this.audioContext.createMediaStreamSource(this.mediaStream);
            analyserSource.connect(this.micAnalyser);
        }
    }
    
    async initDecoder() {
        return new Promise((resolve, reject) => {
            this.decoderWorker = new Worker(resolvePath('assets/decoderWorker.min.js'));
            
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
        return new Promise(async (resolve) => {
            this.miclessMode = false;
            this.mediaStream = null;
            this.recorder = null;
            const mediaDevices = navigator.mediaDevices;
            if (!mediaDevices || !mediaDevices.getUserMedia) {
                console.warn('MediaDevices API unavailable; running mic-less mode.');
                this.miclessMode = true;
                resolve();
                return;
            }
            try {
                // Get mic stream for spectrum analyzer
                this.mediaStream = await mediaDevices.getUserMedia({
                    audio: {
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                    }
                });
            } catch (error) {
                console.warn('Mic permission denied or unavailable, falling back to mic-less mode:', error);
                this.miclessMode = true;
                this.mediaStream = null;
                resolve();
                return;
            }
            try {
                // opus-recorder handles resampling from browser rate to 24kHz internally
                const recorderOptions = {
                    sourceNode: this.audioContext.createMediaStreamSource(this.mediaStream),
                    encoderPath: resolvePath('encoderWorker.min.js'),
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
                    // Send opus data to server
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        const message = new Uint8Array(1 + opusData.length);
                        message[0] = 0x01; // Audio type
                        message.set(new Uint8Array(opusData), 1);
                        this.ws.send(message);
                    }
                };
                
                this.recorder.onstart = () => {
                    console.log('Recorder started');
                };
                
                this.recorder.onstop = () => {
                    console.log('Recorder stopped');
                };
                resolve();
            } catch (error) {
                console.error('Recorder init error:', error);
                this.miclessMode = true;
                this.recorder = null;
                resolve();
            }
        });
    }
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host || 'localhost:8999';
            
            const disabledA = this.isPersonaDisabled('A');
            const disabledB = this.isPersonaDisabled('B');

            const params = new URLSearchParams({
                voice_a: this.elements.voiceA.value,
                voice_b: this.elements.voiceB.value,
                prompt_a: disabledA ? '' : this.elements.promptA.value,
                prompt_b: disabledB ? '' : this.elements.promptB.value,
                disabled_a: disabledA ? '1' : '0',
                disabled_b: disabledB ? '1' : '0',
                mic_disabled: this.miclessMode ? '1' : '0',
            });
            
            const wsPath = `${BASE_PATH}api/conference`;
            const url = `${protocol}//${host}${wsPath}?${params}`;
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
            
            this.ws.onclose = (event) => {
                console.log('WebSocket closed', event.code, event.reason);
                this.updateStatus('disconnected');
                this.isConnected = false;
                this.isRunning = false;
                this.setVoiceSelectorsEnabled(true);
                this.refreshPersonaState('A');
                this.refreshPersonaState('B');
                this.elements.startBtn.style.display = 'inline-block';
                this.elements.stopBtn.style.display = 'none';
                if (event.code === 4001) {
                    alert('Server is currently busy running another conference. Please try again in a moment.');
                }
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
            
            // Track frames for stats
            this.stats.frames++;
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
            disabled_a: this.isPersonaDisabled('A'),
            disabled_b: this.isPersonaDisabled('B'),
        };

        const payload = new TextEncoder().encode(JSON.stringify(config));
        const message = new Uint8Array(1 + payload.length);
        message[0] = 0x0A; // Config type
        message.set(payload, 1);
        this.ws.send(message);
    }

    isPersonaDisabled(persona) {
        const select = persona === 'A' ? this.elements.voiceA : this.elements.voiceB;
        if (!select) return false;
        return select.value === DISABLED_VOICE_VALUE;
    }

    setVoiceSelectorsEnabled(enabled) {
        if (this.elements.voiceA) this.elements.voiceA.disabled = !enabled;
        if (this.elements.voiceB) this.elements.voiceB.disabled = !enabled;
    }

    refreshPersonaState(persona) {
        const disabled = this.isPersonaDisabled(persona);
        const panel = this.personaPanels?.[persona];
        if (panel) {
            panel.classList.toggle('persona-disabled', disabled);
        }
        const promptEl = persona === 'A' ? this.elements.promptA : this.elements.promptB;
        if (promptEl) {
            promptEl.disabled = disabled || this.isRunning;
        }
        const presetContainer = this.presetContainers?.[persona];
        if (presetContainer) {
            const buttons = presetContainer.querySelectorAll('button');
            buttons.forEach((btn) => {
                btn.disabled = disabled || this.isRunning;
            });
        }
    }

    stop() {
        // Stop stats interval
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
            this.statsInterval = null;
        }
        
        // Stop recorder
        if (this.recorder) {
            try { this.recorder.stop(); } catch (e) {}
            this.recorder = null;
        }
        
        // Clean up mic analyser
        this.micAnalyser = null;
        this.micFrequencyData = null;
        
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
        this.analyser = null;
        this.frequencyData = null;
        this.isRunning = false;
        this.isConnected = false;
        this.updateStatus('disconnected');
        this.setVoiceSelectorsEnabled(true);
        this.refreshPersonaState('A');
        this.refreshPersonaState('B');
        
        // Enable download if we have recording data
        if (this.recordingBuffers.mixedAudio.length > 0) {
            this.hasRecording = true;
            this.elements.downloadBtn.disabled = false;
        }
        
        // Reset stats display
        this.levelA = 0;
        this.levelB = 0;
        this.resetStatsDisplay();
        
        this.elements.startBtn.style.display = 'inline-block';
        this.elements.stopBtn.style.display = 'none';
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
    
    async downloadRecording() {
        if (!this.hasRecording || this.recordingBuffers.mixedAudio.length === 0) {
            alert('No recording available');
            return;
        }
        
        // Concatenate all audio chunks
        const totalLength = this.recordingBuffers.mixedAudio.reduce((sum, chunk) => sum + chunk.length, 0);
        const mixedAudio = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of this.recordingBuffers.mixedAudio) {
            mixedAudio.set(chunk, offset);
            offset += chunk.length;
        }
        
        // Create stereo WAV file (mixed audio on both channels)
        // Opus in browser is complex; WAV is universally supported
        console.log('Creating WAV with sample rate:', this.recordingSampleRate, 'samples:', totalLength);
        const wavBlob = this.createWavBlob(mixedAudio, this.recordingSampleRate);
        
        // Use showSaveFilePicker if available, otherwise fallback to download link
        const filename = `dualmind-recording-${new Date().toISOString().slice(0,19).replace(/[:-]/g, '')}.wav`;
        
        if ('showSaveFilePicker' in window) {
            try {
                const handle = await window.showSaveFilePicker({
                    suggestedName: filename,
                    types: [{
                        description: 'WAV Audio',
                        accept: { 'audio/wav': ['.wav'] }
                    }]
                });
                const writable = await handle.createWritable();
                await writable.write(wavBlob);
                await writable.close();
                console.log('Recording saved via File System Access API');
                return;
            } catch (e) {
                if (e.name !== 'AbortError') {
                    console.warn('File picker failed, falling back to download:', e);
                } else {
                    return; // User cancelled
                }
            }
        }
        
        // Fallback: create download link
        const url = URL.createObjectURL(wavBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    createWavBlob(samples, sampleRate) {
        // Create stereo WAV (same audio on both channels)
        const numChannels = 2;
        const bytesPerSample = 2; // 16-bit
        const blockAlign = numChannels * bytesPerSample;
        const byteRate = sampleRate * blockAlign;
        const dataSize = samples.length * numChannels * bytesPerSample;
        const bufferSize = 44 + dataSize;
        
        const buffer = new ArrayBuffer(bufferSize);
        const view = new DataView(buffer);
        
        // WAV header
        const writeString = (offset, str) => {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, bufferSize - 8, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true); // fmt chunk size
        view.setUint16(20, 1, true);  // PCM format
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bytesPerSample * 8, true);
        writeString(36, 'data');
        view.setUint32(40, dataSize, true);
        
        // Write interleaved stereo samples
        let dataOffset = 44;
        for (let i = 0; i < samples.length; i++) {
            const sample = Math.max(-1, Math.min(1, samples[i]));
            const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            // Left channel
            view.setInt16(dataOffset, intSample, true);
            // Right channel (same as left)
            view.setInt16(dataOffset + 2, intSample, true);
            dataOffset += 4;
        }
        
        return new Blob([buffer], { type: 'audio/wav' });
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.conferenceClient = new ConferenceClient();
});
