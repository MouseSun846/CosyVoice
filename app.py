import sys
import os, json, io
from flask_cors import CORS
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import numpy as np
from flask import Flask, request, Response,send_file
import torch
import torchaudio
import lameenc
from cosyvoice.utils.file_utils import load_wav

from pydub import AudioSegment
import torch
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.cli.cosyvoice import CosyVoice2
cosyvoice = CosyVoice2('/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

print(cosyvoice.list_available_spks())
app = Flask(__name__)
CORS(app)  
@app.route("/v1/audio/speech", methods=['POST'])
def stream():
    question_data = request.get_json()
    tts_text = question_data.get('input')
    if not tts_text:
        return {"error": "Query parameter 'query' is required"}, 400
        
    prompt_text = app_config['prompt_text']
    prompt_speech_16k = get_prompt_speech()
    
    def generate_stream():
        mp3_encoder = lameenc.Encoder()
        mp3_encoder.set_bit_rate(128)
        mp3_encoder.set_in_sample_rate(24000)
        mp3_encoder.set_channels(1)
        mp3_encoder.set_quality(2)
        
        try:
         # 直接使用 inference_sft，指定中文女声，启用流式输出
            #for chunk in cosyvoice.inference_sft(tts_text, '中文女', stream=True):
                # 获取音频数据并转换为 PCM
             #   chunk = chunk['tts_speech']
                        
            for chunk in cosyvoice.stream(tts_text, prompt_text, prompt_speech_16k):
                # 直接处理音频数据
                int16_data = (chunk.numpy() * 32767).astype(np.int16)
                
                # 确保编码的数据是 bytes 类型
                mp3_data = mp3_encoder.encode(int16_data.tobytes())
                if mp3_data:
                    yield bytes(mp3_data)  # 确保转换为 bytes
                
            # 处理最后的数据
            mp3_data = mp3_encoder.flush()
            if mp3_data:
                yield bytes(mp3_data)  # 确保转换为 bytes
                
        except Exception as e:
            print(f"Error in generate_stream: {e}")
            yield b''  # 发生错误时返回空字节

    return Response(
        generate_stream(),
        mimetype="audio/mpeg",
        headers={
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        }
    )
# 全局初始化函数
def get_prompt_speech():
    prompt_speech = load_wav('/workspace/CosyVoice/audio/'+app_config['prompt_speech'], 16000)
    prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
    return prompt_speech_16k.float() / (2**15)
if __name__ == "__main__":
    config_path = '/workspace/CosyVoice/audio/config.json'
    if not os.path.exists(config_path):
        print('Config file does not exist.')
        exit(1)
    global app_config
    with open(config_path, 'r', encoding='utf-8') as fr:
        app_config = json.load(fr)
    app.run(host='0.0.0.0', port=50000)
