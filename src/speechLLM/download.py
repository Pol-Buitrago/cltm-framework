# download_models.py
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Processor, Wav2Vec2Model
import torch
import os
import shutil

# Modifica según necesites
speechllm_id = "skit-ai/speechllm-2B"       # o None si no usas SpeechLLM
wav2vec_id = "facebook/wav2vec2-base-960h"  # fallback audio model

out_root = "/tmp/hf_offline_models"  # carpeta local donde guardar todo
os.makedirs(out_root, exist_ok=True)

if speechllm_id:
    print("Descargando SpeechLLM:", speechllm_id)
    # trust_remote_code=True puede ejecutar código remoto; asegúrate de confiar
    tok = AutoTokenizer.from_pretrained(speechllm_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(speechllm_id, trust_remote_code=True)
    # guarda en carpeta local
    local_dir = os.path.join(out_root, speechllm_id.replace("/", "_"))
    os.makedirs(local_dir, exist_ok=True)
    tok.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
    print("Saved SpeechLLM to", local_dir)

print("Descargando wav2vec2 audio model:", wav2vec_id)
proc = Wav2Vec2Processor.from_pretrained(wav2vec_id)
wav2 = Wav2Vec2Model.from_pretrained(wav2vec_id)
local_wav_dir = os.path.join(out_root, wav2vec_id.replace("/", "_"))
os.makedirs(local_wav_dir, exist_ok=True)
proc.save_pretrained(local_wav_dir)
wav2.save_pretrained(local_wav_dir)
print("Saved wav2vec2 to", local_wav_dir)

print("Done. Archiva:", out_root)