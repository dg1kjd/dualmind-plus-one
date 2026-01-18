---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
language:
- en
base_model:
- kyutai/moshiko-pytorch-bf16
extra_gated_prompt: >-
  GOVERNING TERMS: Use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). ADDITIONAL INFORMATION: [CC-BY-4.0](https://huggingface.co/kyutai/moshiko-pytorch-bf16).
---

# PersonaPlex: Voice and role control for full duplex conversational speech models

<style>
h1, h2, h3, h4, h5, h6 {
  color: #76b900; /* NVIDIA green */
  font-weight: 700;
}

hr {
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 2rem 0;
}

/* Improve list spacing */
ul, ol {
  margin-top: 0.5rem;
  margin-bottom: 0.5rem;
}

/* Badge alignment consistency */
img {
  display: inline;
  vertical-align: middle;
}
</style>

➡️ **Code:** [nvidia/personaplex](https://github.com/NVIDIA/personaplex) <br>
➡️ **Demo:** [PersonaPlex Project Page](https://research.nvidia.com/labs/adlr/personaplex/) <br>
➡️ **Paper:** Coming Soon <br>


### Description:
Personaplex is a real-time speech-to-speech conversational model that jointly performs streaming speech understanding and speech generation. The model operates on continuous audio encoded with a neural codec and predicts both text tokens and audio tokens autoregressively to produce its spoken responses. Incoming user audio is incrementally encoded and fed to the model while Personaplex simultaneously generates its own outgoing speech, enabling natural conversational dynamics such as interruptions, barge-ins, overlaps, and rapid turn-taking.
Personaplex runs in a dual-stream configuration in which listening and speaking occur concurrently. This design allows the model to update its internal state based on the user’s ongoing speech while still producing fluent output audio, supporting highly interactive conversations.
Before the conversation begins, Personaplex is conditioned on two prompts: a voice prompt and a text prompt. The voice prompt consists of a sequence of audio tokens that establish the target vocal characteristics and speaking style. The text prompt specifies persona attributes such as role, background, and scenario context. Together, these prompts define the model's conversational identity and guide its linguistic and acoustic behavior throughout the interaction.

This model is ready for commercial use.

### License/Terms of Use:
GOVERNING TERMS: Use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). ADDITIONAL INFORMATION: [CC-BY-4.0](https://huggingface.co/kyutai/moshiko-pytorch-bf16).

### Use Case: <br>
Wherever NVIDIA’s speech-to-speech conversational models are used, PersonaPlex can generate English speech response for English speech input.

### Deployment Geography:
Global

### Release Date:  <br>
Hugging Face [01/15/2026] via [https://huggingface.co/nvidia/personaplex-7b-v1] <br> 
Github [01/15/2026] via [https://github.com/NVIDIA/personaplex] <br> 


## Model Architecture:
**Architecture Type:** Transformer <br>

**Network Architecture:** [Moshi](https://github.com/kyutai-labs/moshi) <br>

Moshi uses:
* Mimi Speech Encoder (ConvNet, Transformer)
* Moshi Temporal Transformer + Depth Transformer
* Mimi Speech Decoder (Transformer, ConvNet)

** This model was developed based on [Moshi (Moshiko weights)](https://huggingface.co/kyutai/moshiko-pytorch-bf16) <br> 
** Number of model parameters: 7B <br>


## Input(s): <br>
**Input Type(s):** Text (prompt), Audio (user speech) <br>
**Input Format:** String, WAV/WebAudio <br>
**Input Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Input:** 24kHz sample rate for audio. <br>

## Output(s)
**Output Type(s):** Text (agent text), Audio (agent speech) <br>
**Output Format:** String, WAV/WebAudio <br>
**Output Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Output:** 24kHz sample rate for audio. <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br> 

## Software Integration:
**Runtime Engine:** PyTorch <br> 

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Ampere (A100)
* NVIDIA Hopper (H100)

**Preferred/Supported Operating System(s):**
* Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>

## Model Version(s):
* v1.0

## Training, Testing, and Evaluation Datasets:

### Training Dataset:
**Link:** Fisher English: [Part1](https://catalog.ldc.upenn.edu/LDC2004S13), [Part2](https://catalog.ldc.upenn.edu/LDC2005S13) <br>
**Data Modality:** Audio (speech) <br>
**Audio Training Data Size:** Less than 10,000 Hours <br>
**Data Collection Method by dataset:** Human <br>
**Labeling Method by dataset:** Automated <br>
**Properties:** 7303 conversations (upto 10 minutes each).


### Testing/Evaluation Dataset:
**Link:** [FullDuplexBench](https://arxiv.org/abs/2503.04721) <br>
**Data Collection Method by dataset:** Hybrid: Human, Synthetic, Automated. <br>
**Labeling Method by dataset:** Automated. <br>
**Properties:** The [FullDuplexBench](https://arxiv.org/abs/2503.04721) public benchmark aggregates various synthetic and real datasets. <br>
Additionally speaker similarity (SSIM) between voice prompts and model outputs on the User Interruption portion of the FullDuplexBench benchmark were measured using [WavLM-TDNN](https://arxiv.org/pdf/2110.13900) embedding cosine similarity.

**FullDuplexBench Benchmark Scores:** <br>
| Metric                                   | Value |
|------------------------------------------|-------|
| Pause Handling(Synthetic): TOR↓          | 0.358 |
| Pause Handling(Candor): TOR↓             | 0.431 |
| Backchannel: TOR↓                        | 0.273 |
| Backchannel: Freq↑                       | 0.042 |
| Backchannel: JSD↓                        | 0.662 |
| Smooth Turn Taking: TOR↑                 | 0.908 |
| Smooth Turn Taking: Latency↓             | 0.170 |
| User Interruption: TOR↑                  | 0.950 |
| User Interruption: GPT-4o↑               | 4.290 |
| User Interruption: Latency↓              | 0.240 |
| User Interruption: SSIM(WavLM)↑          | 0.650 |


**Comparison With Other Conversational AI Systems:**
PersonaPlex outperforms other open-source and commercial systems on conversational dynamics, response and interruption latency, and task adherence in both question-answering assistant and customer service roles.

<figure align="center">
  <img src="figures/results_conversation_dynamics.png" width="1000" />
  <figcaption>
    FullDuplexBench Conversational Dynamics Evaluation. Success rate uses the Takeover Rate (TOR) metric for Smooth Turn-Taking and User Interruption, and 1-TOR for Pause Handling.
  </figcaption>
</figure>

<figure align="center">
  <img src="figures/results_latency.png" width="1000" />
  <figcaption>
    FullDuplexBench Latency Evaluation. Smooth turn-taking latency is measured as the duration from when the user stops speaking to when the agent starts responding. User interruption latency is measured as the duration from when the user interrupts the agent while it is speaking to when the agent stops speaking.
  </figcaption>
</figure>

<figure align="center">
  <img src="figures/results_task_adherence.png" width="1000" />
  <figcaption>
    Task Adherence Evaluation. FullDuplexBench scores are based on general knowledge question-answering in the "User Interruption" category. ServiceDuplexBench (to be released soon) scores are based on varied customer service scenarios. GPT-4o is used to judge the content of agent responses.
  </figcaption>
</figure>

# Inference:
**Acceleration Engine:** PyTorch <br>
**Test Hardware:** NVIDIA A100 80 GB <br>


## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br> 

For more detailed information on ethical considerations for this model, please see the Model Card++ 
[Bias](bias.md), 
[Explainability](explainability.md), 
[Safety & Security](safety.md), 
and [Privacy](privacy.md) Subcards. <br>

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## References(s):
1. [Moshi and Mimi](https://arxiv.org/pdf/2410.00037) <br>
2. [FullDuplexBench](https://arxiv.org/abs/2503.04721) <br>