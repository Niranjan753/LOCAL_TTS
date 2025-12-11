#!/usr/bin/env python3
"""
Example script to run the Dia model with voice cloning (voiceover).
This script uses example_prompt.mp3 as the voice reference.
"""

from pathlib import Path
import soundfile as sf
import numpy as np
import torch
import torchaudio
import tempfile
import os

# Patch torchaudio.load to use soundfile as a workaround for torchcodec issues
_original_torchaudio_load = torchaudio.load

def _patched_torchaudio_load(filepath, **kwargs):
    """Patch torchaudio.load to use soundfile for better compatibility."""
    try:
        # Try using soundfile to load the audio
        data, sr = sf.read(filepath)
        # Convert to torch tensor (channels_first format)
        if len(data.shape) == 1:  # Mono
            audio_tensor = torch.from_numpy(data).unsqueeze(0).float()
        else:  # Stereo or multi-channel
            audio_tensor = torch.from_numpy(data.T).float()
        
        # Handle channels_first parameter
        if kwargs.get('channels_first', False):
            pass  # Already in channels_first format
        else:
            audio_tensor = audio_tensor.transpose(0, 1)
        
        return audio_tensor, sr
    except Exception as e:
        # Fall back to original if soundfile fails
        print(f"Warning: soundfile failed, trying original torchaudio.load: {e}")
        return _original_torchaudio_load(filepath, **kwargs)

# Apply the patch
torchaudio.load = _patched_torchaudio_load

from dia.model import Dia

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = "float16"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = "float32"
else:
    device = torch.device("cpu")
    dtype = "float32"

print(f"Using device: {device}, dtype: {dtype}")

# Load model
print("Loading Dia model...")
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype=dtype, device=device)
print("Model loaded successfully!")

# Use voice cloning for consistent voice (male voice from second half of example_prompt)
example_prompt_path = Path("example_prompt.wav")
if not example_prompt_path.exists():
    example_prompt_path = Path("example_prompt.mp3")

if example_prompt_path.exists():
    print(f"\nExtracting male voice (second half) from {example_prompt_path}...")
    
    # Load the full audio
    audio_data, sample_rate = sf.read(str(example_prompt_path))
    
    # Split in half - take second half (male voice)
    midpoint = len(audio_data) // 2
    male_voice_audio = audio_data[midpoint:]
    
    # Save second half to temporary file
    import tempfile
    temp_male_voice = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_male_voice.name, male_voice_audio, sample_rate)
    male_voice_path = temp_male_voice.name
    
    print(f"✅ Extracted second half ({len(male_voice_audio)/sample_rate:.2f}s) - male voice saved to temporary file")
    
    # Transcript of the SECOND HALF (male voice parts only)
    # Full transcript: "[S1] Open weights... [S2] You get full... [S1] I'm biased... [S2] Hard to disagree. (laughs) [S1] Thanks for listening... [S2] Try it now... [S1] If you liked... [S2] This was Nari Labs."
    # Second half (male voice): from "Thanks for listening" onwards
    clone_from_text = "[S1] Thanks for listening to this demo. [S2] Try it now on Git hub and Hugging Face. [S1] If you liked our model, please give us a star and share to your friends. [S2] This was Nari Labs."
    
    # Your text to generate
    text_to_generate = "[S1] If you are creator or a business owner looking to streamline your entire business process, comment \"CREATOR\" and look out for our message."
    
    # Combine transcript and new text (model needs full text but only generates new part)
    full_text = clone_from_text + "\n" + text_to_generate
    
    print(f"\nUsing male voice for voice cloning...")
    print(f"Generating audio:")
    print(f"{text_to_generate}\n")
    
    output = model.generate(
        full_text,
        audio_prompt=male_voice_path,
        use_torch_compile=False,
        verbose=True,
        cfg_scale=4.0,
        temperature=1.8,
        top_p=0.90,
        cfg_filter_top_k=50,
    )
    
    output_path = "voiceover_example.mp3"
    model.save_audio(output_path, output)
    print(f"\n✅ Voice-cloned audio saved to: {output_path}")
    print("Note: Output contains only your text, but uses the MALE voice from the second half of example_prompt")
    
    # Cleanup temporary file
    import os
    try:
        os.unlink(male_voice_path)
    except:
        pass
else:
    # Fallback: generate without voice cloning
    print("\nNote: example_prompt.wav/mp3 not found. Generating without voice cloning...")
    text = "[S1] If you are creator or a business owner looking to streamline your entire business process, comment \"CREATOR\" and look out for our message."
    
    print(f"\nGenerating audio for:")
    print(f"{text}\n")
    
    output = model.generate(
        text,
        use_torch_compile=False,
        verbose=True,
        cfg_scale=3.0,
        temperature=1.8,
        top_p=0.90,
        cfg_filter_top_k=50,
    )
    
    output_path = "voiceover_example.mp3"
    model.save_audio(output_path, output)
    print(f"\n✅ Audio saved to: {output_path}")

