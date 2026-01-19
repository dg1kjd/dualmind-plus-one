# DualMind+1 Moshi Edition

**DualMind+1 Moshi Edition** (or **DualMind+1**) is a cutting-edge system derived from NVIDIA's PersonaPlex, enhanced and customized for advanced audio generation and conferencing capabilities. This project integrates various components to facilitate real-time audio processing and model inference, leveraging powerful machine learning models.

## Overview

DualMind+1 combines elements from NVIDIA's original framework with custom client and conference UI implementations. It utilizes the `moshi` inference codebase for Kyutai audio generation models, adapted from Audiocraft by Meta Platforms, to deliver high-quality audio interactions. **This is a derivative work of NVIDIA's PersonaPlex system and model**, building upon their foundational technology. For original documentation and details, refer to `README_nv.md` in this repository.

## System Components

- **Client**: A React-based frontend for user interaction, handling audio input and output.
- **Conference UI**: A web interface for real-time audio conferencing, supported by custom JavaScript for audio processing.
- **Moshi**: The core inference engine for audio generation, based on Kyutai's models.
- **Weights**: Pre-trained model weights for audio processing, under NVIDIA Open Model License. **Note**: Weights are not distributed with this source code; they are downloaded from Hugging Face on first start.

## Requirements

- **Hardware**: Recommended 2 CUDA devices, minimum 2x RTX 3090. Tested on 1x RTX 3090 and 1x RTX 5090.
- **Software**: Python 3.8+, CUDA toolkit compatible with your GPU.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dualmind-plus-one.git
   cd dualmind-plus-one
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -e moshi
   # Additional dependencies may be required based on your setup
   ```

## Running the System

To start the DualMind+1 conference system, use the following command:
```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python -m moshi.conference --device-a cuda:0 --device-b cuda:1 --static conference_ui --port 8999
```

- `--device-a` and `--device-b`: Specify the CUDA devices for model inference (e.g., `cuda:0` and `cuda:1`).
- `--static`: Points to the directory containing static files for the conference UI.
- `--port`: The port on which the server will run (default: 8999).

Ensure you have at least two powerful GPUs available for optimal performance, tested on 1xRTX5090 and 1xRTX3090, but 2xRTX3090 should be just fine.
VRAM consumption: ~19.5GB per GPU.

## License

This project is released under the MIT License. See `LICENSE-MIT` for details. Note that model weights are under the NVIDIA Open Model License, as described in `README_nv.md`.

**Copyright 2026 Jens David Consulting**

**Authors**: Jens David (JDC); Opus-4.5, Claude (Anthropic)

## Disclaimer

**IMPORTANT DISCLAIMER**: This software and all associated materials are provided "AS IS" and "WITH ALL FAULTS," without any warranties or conditions of any kind, whether express, implied, or statutory, including but not limited to warranties of merchantability, fitness for a particular purpose, title, non-infringement, or any other warranty. The entire risk as to the quality and performance of the software is with you. In no event shall the authors, contributors, or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software. Use at your own risk. This project is a derivative work and does not imply endorsement by NVIDIA or any other original contributors.

## Acknowledgments

- Original codebase derived from NVIDIA's PersonaPlex.
- `moshi` inference code adapted from Kyutai and Meta Platforms' Audiocraft.
