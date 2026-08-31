FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/runpod-volume/models \
    HF_HOME=/runpod-volume/models/huggingface \
    MODELSCOPE_CACHE=/runpod-volume/models/modelscope \
    PYTHONPATH=/app/vendor/thonburian-tts:/app/vendor/thonburian-tts/src:/app/vendor/CosyVoice:/app/vendor/CosyVoice/third_party/Matcha-TTS \
    JAITTS_CHECKPOINT=hf://gostiktok14sroy/gostts/model.pt \
    JAITTS_VOCAB=hf://gostiktok14sroy/gostts/vocab.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs build-essential libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.3.1+cu121" "torchaudio==2.3.1+cu121"
RUN python -m pip install -r /app/requirements.txt
# F5-TTS declares UI/training dependencies and unpinned audio wheels that can
# replace the CUDA-matched Torch stack. The worker only needs its inference code.
RUN python -m pip install --no-deps "f5-tts==1.1.22"
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps "torch==2.3.1+cu121" "torchaudio==2.3.1+cu121"
RUN python -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('torchaudio', torchaudio.__version__)"

RUN mkdir -p /app/vendor \
    && git clone https://github.com/biodatlab/thonburian-tts.git /app/vendor/thonburian-tts \
    && git -C /app/vendor/thonburian-tts checkout 032fe7e51674afe066a98e6d3cf47fc96d04b290 \
    && git clone https://github.com/FunAudioLLM/CosyVoice.git /app/vendor/CosyVoice \
    && git -C /app/vendor/CosyVoice checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc \
    && git -C /app/vendor/CosyVoice submodule update --init --recursive \
    && python -m pip install --no-deps -e /app/vendor/thonburian-tts \
    && python -c "import flowtts; from flowtts.load_flowtts import FlowTTS; print('flowtts import ok', FlowTTS)"

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
