/**
 * Conference Audio Playback Worklet
 * Handles decoded audio playback from server
 * 
 * Audio flow:
 * - Decoder worker decodes Opus → sends Float32Array PCM
 * - This worklet buffers and plays at browser's native sample rate
 * - opus-recorder handles mic separately (no input processing here)
 */

class ConferenceProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // Ring buffer for efficient audio buffering (avoids O(n) shift operations)
        this.ringBuffer = new Float32Array(48000);  // 1 second at 48kHz
        this.writePos = 0;
        this.readPos = 0;
        this.bufferedSamples = 0;
        
        // Start playback with minimal latency - just 2 worklet frames (256 samples = ~5ms)
        this.initialBufferSize = 256;
        this.started = false;
        // Don't stop playback until buffer is truly empty for sustained period
        this.emptyFrameCount = 0;
        this.maxEmptyFrames = 3;  // Allow 3 empty frames (~8ms) before stopping
        
        console.log(`ConferenceProcessor: sampleRate=${sampleRate}, initialBuffer=${this.initialBufferSize}`);
        
        this.receiveCount = 0;
        this.port.onmessage = (event) => {
            if (event.data.type === 'audio') {
                this.receiveCount++;
                // Received decoded PCM samples from decoder worker
                const samples = event.data.samples;
                if (samples && samples.length > 0) {
                    // Write samples to ring buffer
                    const len = samples.length;
                    const bufLen = this.ringBuffer.length;
                    
                    for (let i = 0; i < len; i++) {
                        this.ringBuffer[this.writePos] = samples[i];
                        this.writePos = (this.writePos + 1) % bufLen;
                    }
                    this.bufferedSamples += len;
                    
                    // Cap buffered count to prevent overflow tracking
                    if (this.bufferedSamples > bufLen) {
                        this.bufferedSamples = bufLen;
                    }
                    
                    // Start playback when we have enough data
                    if (!this.started && this.bufferedSamples >= this.initialBufferSize) {
                        this.started = true;
                        this.emptyFrameCount = 0;
                        console.log('[WORKLET] Playback started, buffered:', this.bufferedSamples);
                    }
                    
                    // Log receive stats periodically
                    if (this.receiveCount <= 5 || this.receiveCount % 100 === 0) {
                        console.log('[WORKLET] Received audio, count:', this.receiveCount, 'buffered:', this.bufferedSamples);
                    }
                }
            } else if (event.data.type === 'reset') {
                this.writePos = 0;
                this.readPos = 0;
                this.bufferedSamples = 0;
                this.started = false;
                this.emptyFrameCount = 0;
            }
        };
    }
    
    process(inputs, outputs, parameters) {
        const output = outputs[0];
        if (!output || output.length === 0) {
            return true;
        }
        
        const frameSize = output[0].length;
        const bufLen = this.ringBuffer.length;
        
        if (this.started && this.bufferedSamples > 0) {
            const samplesToPlay = Math.min(frameSize, this.bufferedSamples);
            
            // Read from ring buffer efficiently
            for (let i = 0; i < samplesToPlay; i++) {
                const sample = this.ringBuffer[this.readPos];
                this.readPos = (this.readPos + 1) % bufLen;
                
                // Write to all output channels (handles stereo)
                for (let ch = 0; ch < output.length; ch++) {
                    output[ch][i] = sample;
                }
            }
            this.bufferedSamples -= samplesToPlay;
            
            // Fill rest with silence
            for (let i = samplesToPlay; i < frameSize; i++) {
                for (let ch = 0; ch < output.length; ch++) {
                    output[ch][i] = 0;
                }
            }
            
            // Track empty frames - only stop after sustained silence
            if (this.bufferedSamples === 0) {
                this.emptyFrameCount++;
                if (this.emptyFrameCount >= this.maxEmptyFrames) {
                    this.started = false;
                    console.log('[WORKLET] Playback paused - buffer empty');
                }
            } else {
                this.emptyFrameCount = 0;
            }
        } else {
            // Output silence while waiting
            for (let ch = 0; ch < output.length; ch++) {
                for (let i = 0; i < frameSize; i++) {
                    output[ch][i] = 0;
                }
            }
        }
        
        return true;
    }
}

registerProcessor('conference-processor', ConferenceProcessor);
