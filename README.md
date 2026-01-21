<p align="center">
  <img src="conference_ui/jdc.png" alt="DualMind+1 Moshi Edition logo" width="180" />
</p>

<h1 align="center">DualMind+1 Moshi Edition</h1>

<p align="center">
  <strong>Full-duplex, sub-250ms, voice-to-voice AI conversations derived from NVIDIA's PersonaPlex.</strong><br/>
  <a href="https://github.com/dg1kjd/dualmind-plus-one">GitHub Repository</a> ·
  <a href="https://jens-david-consulting.com/dualmind/">Live Demo</a>
</p>

---

### Quick Links

- 🔗 **Repository**: [github.com/dg1kjd/dualmind-plus-one](https://github.com/dg1kjd/dualmind-plus-one)
- 🚀 **Live Demo**: [https://jens-david-consulting.com/dualmind/](https://jens-david-consulting.com/dualmind/)

---

## Contents

1. [Overview](#overview)
2. [System Components](#system-components)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Running the System](#running-the-system)
6. [Credits / License / Authors](#credits--license--copyright--authors)
7. [Disclaimer](#disclaimer)
8. [Acknowledgments](#acknowledgments)

---

## Overview

**DualMind+1 Moshi Edition** (or **DualMind+1**) is a production-re^WAI-slo^W voice-to-voice full duplex conversational conference system derived from NVIDIA's PersonaPlex (which in turn is derived from Kyutai Moshi). Basically two instances of the PersonaPlex model running in parallel, talking to each other, and (optionally) the user as well. It therefore showcases what these audio-based models are capable off: Full human-like conversation flow with extremely low (sub-250ms) latency, full-duplex operation, barge in, backgrounding (um-humming, etc., "right", "ok", "yes", "sure", "that's right"), exclamations ("gosh!", "oh my god", laughter), stuttering, coughing, and lots of "ummm" and "ahhhs". It is the audio version of the well-known DualMind system that lets two text-based LLMs talk to each other.
In one sentence: The output that this demo produces feels like a natural conversation between two humans, which at times is very impressive.
Sysprompt-like text descriptions for both AI parties can be provided. Several different voices, both native and with non-native absolutely real-sounding accents can be selected for each AI party individually. By disabling one AI party (setting AI party B voice to "None/Disabled") you basically get the functionality of NVIDIA's PersonaPlex system, i.e. you can talk to a single AI party using your mic/speaker. Headset not required in any case, due to echo cancellation and intrinsic echo resistence of the models. Language: Only English trained. Intelligence: If you are looking for a highly intelligent AI assistant, this is not the right tool for you. DualMind+1 is a -- possibly entertaining -- conversational system, not a chatbot. It is designed to simulate human conversation, not to provide intellectually stimulating or scientifically/technically deep/accurate insights. Neither is there "a simple way" to make it "smart" like Grok/Claude or even Gemma by "wiring in" a text-based LLM because it operates on audio instead of text.

The system consists of a server part and a client part. The server part comprises a web server that provides via static HTTP server the conference UI client and handles the audio conferencing / signal processing and neural network processing via PyTorch / modified Moshi framework. It requires and automatically downloads NVIDIA's PersonaPlex weights from Hugging Face on first start.
The client part is a React-based frontend for user interaction, handling audio input and output, served to a standard web browser (must be pretty recent for WASM, Opus, Microphone access, Websocket etc.) via the built-in static HTTP server. It can work either locally or via the Internet, leveraging OPUS audio compression for reduced bandwidth usage and also provides transcriptions of the audio streams.
The backend part of the server uses PyTorch to run inference on the two Moshi instances and Mimi encoders and decoders, as well as light DSP (channel mixing, resampling, etc.).

## System Components

- **Client Frontend**: A React-based frontend for user interaction, handling audio input and output. Uses WASM and websocket for audio handling and transmission. Opus audio compression is used for reduced bandwidth usage.
- **Conference UI**: A web interface for real-time audio conferencing, supported by some JavaScript plumbing. Spectrum estimator for AI parties and user microphone.
- **Moshi**: The core inference engine for audio generation, based on Kyutai's models, running NVIDIA's PersonaPlex weights, slightly modified for performance and Blackwell support.
- **Mimi**: The neural audio codec using Residual Vector Quantization (RVQ) for datarate-reduced model I/O instead of tokenization.
- **Weights**: Pre-trained model weights for audio processing, under NVIDIA Open Model License. **Note**: Weights are not distributed with this source code; they are downloaded from Hugging Face on first start.

## Requirements

- **Hardware**: Recommended 2 CUDA devices, minimum 2x RTX 3090. Tested on 1x RTX 3090 and 1x RTX 5090. VRAM consumption: ~19.5GB per GPU. Two RTX 3090s should be just fine as well. Alternatively data center-grade compute (i.e. A100/H100). The system runs one moshi instance per GPU together with its respective mimi codecs. CPU is not used heavily, only for mixing and simple SRC.
- **Software**: Tested with Python 3.11.12, Pytorch nightly (torch-2.11.0.dev20260117+cu128), CUDA 12.8, Transformers 4.57.3. Confer requirements.txt for rough guidance.
- **Power Consumption**: About 240W for RTX 3090, 190W for RTX 5090 (continuous during conversation)
- **Hugging Face account** for downloading weights.
- **Network**: Should be as low-latency as possible, audio buffering is set to 80ms. LAN/localhost connection recommended.
- **Audio**: Microphone/Speaker combo, headset optional.
- **Recent Browser**: Recent Chrome, Firefox, Safari, iPhone Safari. Edge, Android not tested. Uses no WebRTC.
- **Optional SSL Certs**: If non-local usage is desired, SSL certificates are required because browser will refuse audio input/output if the connection is not secure.
- **Optional Inbound Proxy**: If non-local usage is desired, an inbound proxy is recommended for security and sanitization. Recent Apache with mod_proxy and mod_proxy_wstunnel is sufficient and proved performant for single user inference.

## Installation

This is an experimental system, these are only approximate instructions. You should know what you are doing w/r/t Pytorch dependencies and CUDA.

1. Clone the repository:
   ```bash
   git clone https://github.com/dg1kjd/dualmind-plus-one.git
   cd dualmind-plus-one
   ```
2. Set up a virtual environment for all backend components:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install a CUDA 12.8+/sm_120 capable PyTorch nightly build (required for RTX 5090 / Blackwell class GPUs). If you run older GPUs you may choose a different wheel, but the nightly ensures kernels exist for the latest architectures:
   ```bash
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```
4. Install the remaining backend dependencies (this also installs the local `moshi` package via the editable requirement):
   ```bash
   pip install -r requirements.txt
   ```
5. Build the React client (note the `client/` subfolder):
   ```bash
   cd client
   npm install
   npm run build
   ```
   The build artifacts will be emitted to `client/dist/`; the server can also serve the checked-in `conference_ui` folder.
6. Generate self-signed SSL certificate (if needed). **Note: SSL required for audio to work**
   ```bash
   apt-get update && apt-get install -y openssl
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout key.pem -out cert.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
   ```

Tested on heavily modified Ubuntu 24.04.6 LTS with CUDA 12.8 and Consumer Blackwell & Ampere.

## Running the System

To start the DualMind+1 conference system, use the following command:
```bash
source .venv/bin/activate && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
PYTHONUNBUFFERED=1 python -m moshi.conference \
  --device-a cuda:0 --device-b cuda:1 \
  --static conference_ui --port 8999
```
Will run locally, point web browser to https://localhost:8999 . ***The "S" is important because most web browsers will refuse to access the microphone if the connection is not secure.***

Or, if running publicly with inbound proxy:
```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python -m moshi.conference --device-a cuda:0 --device-b cuda:1 --static conference_ui --port 8999 --allowed-origin https://www.your-origin-here.com
```

- `--device-a` and `--device-b`: Specify the CUDA devices for model inference (e.g., `cuda:0` and `cuda:1`).
- `--static`: Points to the directory containing static files for the conference UI.
- `--port`: The port on which the server will run (default: 8999).

## Credits / License / Copyright / Authors

**This is a derivative work of NVIDIA's PersonaPlex system and -model and Kyutai's Moshi**. For original documentation and details, refer to `README_nv.md` and Moshi docs in this repository. Their respective licenses and copyrights apply.

This project is released under the MIT License. See `LICENSE-MIT` for details. Note that model weights are under the NVIDIA Open Model License, as described in `README_nv.md`.

**Copyright 2026 Jens David Consulting (derivative work only)**
**Authors**: David, Jens (JDC) <dm2026@jens-david-consulting.com> @dg1kjd on X // Opus-4.5, Claude (Anthropic)
**Original Authors**: NVIDIA, Kyutai, see respective docs

## Disclaimer

**IMPORTANT DISCLAIMER**: This software and all associated materials are provided "AS IS" and "WITH ALL FAULTS," without any warranties or conditions of any kind, whether express, implied, or statutory, including but not limited to warranties of merchantability, fitness for a particular purpose, title, non-infringement, or any other warranty. The entire risk as to the quality and performance of the software is with you. In no event shall the authors, contributors, or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software. Use at your own risk. This project is a derivative work and does not imply endorsement by NVIDIA or any other original contributors.
DO NOT USE IT FOR BAD STUFF PLEASE.

## Acknowledgments

- Original codebase derived from NVIDIA's PersonaPlex. Model: NVIDIA
- `moshi` inference code adapted from Kyutai
- mimi codec: Kyutai