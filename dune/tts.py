"""Server-side Text-to-Speech with eccentric scientist voice personality."""

import pyttsx3
import os
from typing import Optional

class ECCENTRICVoiceTTS:
    """TTS engine configured for DUNE's eccentric scientist personality."""
    
    def __init__(self, output_dir: str = "ui/audio"):
        self.engine = pyttsx3.init()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self._configure_voice()
    
    def _configure_voice(self):
        """Configure voice with eccentric scientist parameters."""
        # Speed: 1.1-1.3x (pyttsx3 rate is in words per minute, default ~200)
        self.engine.setProperty('rate', 240)  # ~1.2x speed
        
        # Pitch: variable (slightly higher for eccentric tone)
        self.engine.setProperty('pitch', 1.15)
        
        # Volume
        self.engine.setProperty('volume', 0.95)
        
        # Try to use a voice with character (male/female preference can vary)
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            # Use the second voice if available (often has more character)
            self.engine.setProperty('voice', voices[1].id)
    
    def _inject_personality(self, text: str) -> str:
        """Add eccentric scientist interjections and pauses to text."""
        # Add occasional "Ah", "Hmm", "Interesting" at natural break points
        import random
        
        sentences = text.split('. ')
        result = []
        
        for i, sentence in enumerate(sentences):
            if i > 0 and random.random() < 0.3:  # 30% chance on subsequent sentences
                interjection = random.choice(['Ah, ', 'Hmm, ', 'Interesting. '])
                result.append(interjection + sentence)
            else:
                result.append(sentence)
        
        text_with_personality = '. '.join(result)
        return text_with_personality
    
    def synthesize(self, text: str, filename: Optional[str] = None) -> str:
        """
        Convert text to speech and save to file.
        
        Args:
            text: Text to synthesize
            filename: Output filename (auto-generated if not provided)
            
        Returns:
            Path to generated audio file
        """
        # Add personality interjections
        text_with_personality = self._inject_personality(text)
        
        # Generate filename if not provided
        if not filename:
            import hashlib
            import time
            hash_suffix = hashlib.md5(text.encode()).hexdigest()[:8]
            filename = f"dune_{hash_suffix}_{int(time.time())}.mp3"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Synthesize and save
        self.engine.save_to_file(text_with_personality, filepath)
        self.engine.runAndWait()
        
        return filepath
    
    def get_audio_url(self, filepath: str) -> str:
        """Convert file path to web-accessible URL."""
        # Assuming audio files are served from /audio/ endpoint
        filename = os.path.basename(filepath)
        return f"/audio/{filename}"


# Singleton instance
_tts_engine = None

def get_tts() -> ECCENTRICVoiceTTS:
    """Get or create the TTS engine."""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = ECCENTRICVoiceTTS()
    return _tts_engine

def synthesize_response(text: str) -> dict:
    """
    Synthesize a response and return metadata.
    
    Returns:
        {
            'text': original_text,
            'audio_url': web_accessible_url,
            'audio_file': local_filepath
        }
    """
    tts = get_tts()
    audio_file = tts.synthesize(text)
    audio_url = tts.get_audio_url(audio_file)
    
    return {
        'text': text,
        'audio_url': audio_url,
        'audio_file': audio_file
    }
