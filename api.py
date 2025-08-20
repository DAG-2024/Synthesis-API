from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import soundfile as sf
import numpy as np
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import signal
import threading
import time
import uuid
from typing import Optional
import tempfile

app = FastAPI(title="DAG Text-to-Speech API", description="API for generating speech from text using DAG model")

# Resolve static directory relative to this file, regardless of current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static files at /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global variables for models
tokenizer = None
model = None
Codec_model = None

def load_models():
    """Load the required models"""
    global tokenizer, model, Codec_model
    
    print("Loading models...")
    
    # Load LLaSA-3B model
    llasa_3b = 'srinivasbilla/llasa-3b'
    tokenizer = AutoTokenizer.from_pretrained(llasa_3b)
    model = AutoModelForCausalLM.from_pretrained(llasa_3b)
    model.eval()
    model.to('cuda')
    
    # Load XCodec2 model with retries and deferred import so failures don't crash startup
    def _try_load_codec_model() -> bool:
        try:
            from xcodec2.modeling_xcodec2 import XCodec2Model  # local import to avoid hard dependency at import time
            model_path = "srinivasbilla/xcodec2"
            codec = XCodec2Model.from_pretrained(model_path)
            codec.eval().cuda()
            globals()["Codec_model"] = codec
            return True
        except Exception as err:
            print(f"XCodec2 load attempt failed: {err}")
            return False

    Codec_model = None
    max_attempts = int(os.getenv("CODEC_LOAD_MAX_RETRIES", "1"))
    retry_delay_s = float(os.getenv("CODEC_LOAD_RETRY_SECONDS", "5"))
    for attempt_idx in range(max_attempts):
        if _try_load_codec_model():
            print("XCodec2 model loaded successfully")
            break
        if attempt_idx < max_attempts - 1:
            print(f"Retrying XCodec2 load in {retry_delay_s} seconds (attempt {attempt_idx+2}/{max_attempts})...")
            time.sleep(retry_delay_s)
    if Codec_model is None:
        print("Warning: XCodec2 model unavailable after initial attempts. Background retries may continue.")


def _retry_load_codec_model_background():
    """Background worker to keep retrying loading the XCodec2 model until success or max attempts reached."""
    max_attempts = int(os.getenv("CODEC_LOAD_BG_MAX_RETRIES", "20"))
    retry_delay_s = float(os.getenv("CODEC_LOAD_BG_RETRY_SECONDS", "15"))
    for attempt_idx in range(max_attempts):
        try:
            from xcodec2.modeling_xcodec2 import XCodec2Model
            model_path = "srinivasbilla/xcodec2"
            codec = XCodec2Model.from_pretrained(model_path)
            codec.eval().cuda()
            globals()["Codec_model"] = codec
            print(f"XCodec2 model loaded successfully in background on attempt {attempt_idx+1}")
            return
        except Exception as err:
            print(f"Background retry {attempt_idx+1}/{max_attempts} failed to load XCodec2: {err}")
            time.sleep(retry_delay_s)
    print("Background retries exhausted; XCodec2 model not loaded.")
    
    print("Models loaded successfully!")

def resample_audio(audio, orig_sr, new_sr):
    """Resample audio to target sample rate"""
    if orig_sr == new_sr:
        return audio
    # Calculate the ratio
    ratio = new_sr / orig_sr
    # Calculate new length
    new_length = int(len(audio) * ratio)
    # Create time arrays
    old_time = np.arange(len(audio)) / orig_sr
    new_time = np.arange(new_length) / new_sr
    # Linear interpolation
    resampled = np.interp(new_time, old_time, audio)
    return resampled

def ids_to_speech_tokens(speech_ids):
    """Convert speech IDs to token format"""
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
    """Extract speech IDs from token format"""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def process_audio_file(audio_file_path: str):
    """Process audio file to get prompt waveform"""
    try:
        # Try using torchaudio first
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        # Check duration and trim if needed
        if len(waveform[0])/sample_rate > 15:
            print("Trimming audio to first 15 seconds.")
            waveform = waveform[:, :sample_rate*15]
        
        # Convert stereo to mono if needed
        if waveform.size(0) > 1:
            waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform_mono = waveform
        
        # Resample to 16kHz
        prompt_wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_mono)
        
    except Exception as e:
        print(f"Torchaudio failed, trying soundfile: {e}")
        # Fallback to soundfile
        prompt_wav, sr = sf.read(audio_file_path)
        
        # Convert to mono first
        if len(prompt_wav.shape) > 1 and prompt_wav.shape[0] > 1:
            prompt_wav = np.mean(prompt_wav, axis=0)
        
        # Resample to 16kHz
        if sr != 16000:
            prompt_wav = resample_audio(prompt_wav, sr, 16000)
            sr = 16000
        
        # Check duration and trim if needed
        duration_seconds = len(prompt_wav) / sr
        if duration_seconds > 15:
            print("Trimming audio to first 15 seconds.")
            max_samples = int(15 * sr)
            prompt_wav = prompt_wav[:max_samples]
        
        # Convert to PyTorch tensor
        prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)
    
    return prompt_wav

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()
    # If codec failed to load, keep trying in background without blocking the server
    if Codec_model is None:
        threading.Thread(target=_retry_load_codec_model_background, daemon=True).start()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serves the HTML interface"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/generate-speech")
async def generate_speech(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Input audio file (max 15 seconds)"),
    input_transcription: str = Form(..., description="Input transcription of the audio"),
    target_text: str = Form(..., description="Target text to generate speech for")
):
    """
    Generate speech from target text using input audio as reference
    
    - **audio_file**: Input audio file (WAV format, max 15 seconds)
    - **input_transcription**: Transcription of the input audio
    - **target_text**: Target text to generate speech for
    """
    
    # Ensure required models are available
    if Codec_model is None:
        raise HTTPException(status_code=503, detail="Audio codec model is unavailable (still initializing). Try again shortly.")

    # Validate file type
    if not audio_file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        input_path = os.path.join(temp_dir, "input.wav")
        with open(input_path, "wb") as buffer:
            buffer.write(await audio_file.read())
        
        # Generate unique output filename
        output_filename = f"generated_{uuid.uuid4().hex}.wav"
        # Create a persistent temp file for the response, so it survives after this context manager
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Process audio file
            prompt_wav = process_audio_file(input_path)
            
            # Prepare input text
            input_text = input_transcription + target_text
            
            # Generate speech
            with torch.no_grad():
                # Encode the prompt wav
                vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
                print("Prompt Vq Code Shape:", vq_code_prompt.shape)
                
                vq_code_prompt = vq_code_prompt[0, 0, :]
                # Convert int 12345 to token <|s_12345|>
                speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)
                
                formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
                
                # Tokenize the text and the speech prefix
                chat = [
                    {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                    {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
                ]
                
                input_ids = tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    return_tensors='pt',
                    continue_final_message=True
                )
                input_ids = input_ids.to('cuda')
                speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
                
                # Generate the speech autoregressively
                outputs = model.generate(
                    input_ids,
                    max_length=2048,
                    eos_token_id=speech_end_id,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.6,
                )
                
                # Extract the speech tokens
                generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
                
                speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Convert token <|s_23456|> to int 23456
                speech_tokens = extract_speech_ids(speech_tokens)
                
                speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
                # Decode the speech tokens to speech waveform
                gen_wav = Codec_model.decode_code(speech_tokens)
                
                # Get only the generated part
                gen_wav = gen_wav[:, :, prompt_wav.shape[1]:]
                
                # Save the generated audio
                sf.write(output_path, gen_wav[0, 0, :].cpu().numpy(), 16000)
            
            # Return the generated audio file
            # Schedule deletion of the temp file after response is sent
            background_tasks.add_task(os.remove, output_path)
            return FileResponse(path=output_path, media_type="audio/wav", filename=output_filename)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "tokenizer": tokenizer is not None,
            "model": model is not None,
            "codec_model": Codec_model is not None
        }
    }

@app.get("/api-info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "DAG Text-to-Speech API",
        "version": "1.0.0",
        "description": "API for generating speech from text using DAG model",
        "endpoints": {
            "/": "HTML interface",
            "/generate-speech": "Generate speech from audio and text",
            "/health": "Health check",
            "/kill": "Shutdown server",
            "/docs": "API documentation (Swagger UI)",
            "/redoc": "API documentation (ReDoc)"
        },
        "requirements": {
            "input_audio": "WAV format, max 15 seconds",
            "input_transcription": "Text transcription of input audio",
            "target_text": "Text to generate speech for"
        }
    }

@app.get("/kill")
async def kill():
    """Shutdown the server process gracefully"""
    def _shutdown_after_response():
        # Delay shutdown to allow response to be sent
        import time
        time.sleep(0.2)
        # Send SIGTERM to allow graceful shutdown
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_shutdown_after_response, daemon=True).start()
    return {"status": "shutting down"}


if __name__ == "__main__":
    import uvicorn
    print("starting server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
