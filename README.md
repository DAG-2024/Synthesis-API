# DAG Text-to-Speech API

This API provides text-to-speech generation capabilities using the DAG (Direct Audio Generation) model. It takes an input audio file (max 15 seconds), input transcription, and target text to generate a new audio file.

## Features

- **Input Processing**: Accepts WAV audio files up to 15 seconds
- **Audio Resampling**: Automatically resamples audio to 16kHz
- **Stereo to Mono Conversion**: Handles both stereo and mono input files
- **Speech Generation**: Uses LLaSA-3B model for speech generation
- **Audio Encoding/Decoding**: Uses XCodec2 model for audio processing

## Requirements

- Python 3.9
- CUDA-compatible GPU
- At least 16GB GPU memory

## Installation

1. **Create and activate a conda environment** (recommended):
   ```bash
   conda create -n DAG_API python=3.9 -y
   conda activate DAG_API
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify CUDA installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Usage

### Starting the API Server

```bash
cd DAG
./start_api.sh
```

The server will start on `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```
Returns the status of the API and loaded models.

#### 2. Generate Speech
```bash
POST /generate-speech
```

**Parameters**:
- `audio_file` (file): Input WAV audio file (max 15 seconds)
- `input_transcription` (string): Transcription of the input audio
- `target_text` (string): Target text to generate speech for

**Response**: Generated WAV audio file

### Example Usage

#### Using curl:
```bash
curl -X POST "http://localhost:8000/generate-speech" \
  -F "audio_file=@audio1.wav" \
  -F "input_transcription=things escalate a bit. So how do you represent letters? Because obviously this makes our devices more useful, whether it's in English or any other human language. How could we go about representing the letter A for instance?" \
  -F "target_text=If at the end of the day, all our computers, all our phones have access to is electricity or equivalently switches or metaphorically tiny little light bulbs inside of them that can be on and off." \
  --output generated_speech.wav
```

#### Using Python:
```python
import requests

url = "http://localhost:8000/generate-speech"
files = {
    'audio_file': ('audio1.wav', open('audio1.wav', 'rb'), 'audio/wav')
}
data = {
    'input_transcription': 'Your input transcription here',
    'target_text': 'Your target text here'
}

response = requests.post(url, files=files, data=data)
if response.status_code == 200:
    with open('output.wav', 'wb') as f:
        f.write(response.content)
```

#### Using the test script:
```bash
python test_api.py
```

### API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Model Information

- **LLaSA-3B**: Text-to-speech generation model
- **XCodec2**: Audio encoding/decoding model
- **Sample Rate**: 16kHz
- **Max Input Duration**: 15 seconds

## Error Handling

The API includes comprehensive error handling for:
- Invalid file formats
- Audio processing errors
- Model loading issues
- CUDA/GPU errors

## Performance Notes

- First request may take longer due to model loading
- GPU memory usage depends on input audio length
- Processing time scales with target text length

## Troubleshooting

1. **CUDA out of memory**: Reduce input audio length or use smaller models
2. **Model loading fails**: Check internet connection and available disk space
3. **Audio processing errors**: Ensure input is a valid WAV file

