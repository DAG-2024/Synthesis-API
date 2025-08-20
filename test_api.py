import requests
import os

def test_api():
    """Test the DAG API with the existing audio file"""
    
    # API endpoint
    url = "http://localhost:8000/generate-speech"
    
    # Test data
    input_transcription = "things escalate a bit. So how do you represent letters? Because obviously this makes our devices more useful, whether it's in English or any other human language. How could we go about representing the letter A for instance?"
    target_text = "Fix implemented: The server now writes the WAV to a persistent temp file."
    
    # Check if sample_audio.wav exists
    audio_file_path = "./DAG/sample_audio.wav"
    if not os.path.exists(audio_file_path):
        print(f"Error: {audio_file_path} not found!")
        return
    
    # Prepare the request
    files = {
        'audio_file': ('sample_audio.wav', open(audio_file_path, 'rb'), 'audio/wav')
    }
    
    data = {
        'input_transcription': input_transcription,
        'target_text': target_text
    }
    
    try:
        print("Sending request to API...")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            # Save the generated audio
            output_filename = "api_generated_output.wav"
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            print(f"Success! Generated audio saved as: {output_filename}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api() 