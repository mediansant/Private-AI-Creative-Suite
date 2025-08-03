import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import gc
import os

class LocalAudioGenerator:
    """
    Local audio generation using Meta's MusicGen.
    Runs entirely on your device for complete privacy.
    """
    
    def __init__(self, model_size="medium"):
        """
        Initialize the audio generator with MusicGen.
        
        Args:
            model_size (str): Model size - "small", "medium", or "large"
        """
        self.model_size = model_size
        self.model = None
        self.processor = None
        self.device = self._get_device()
        self._load_model()
    
    def _get_device(self):
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        else:
            return "cpu"
    
    def _get_model_id(self, size):
        """Get the appropriate model ID based on size."""
        model_ids = {
            "small": "facebook/musicgen-small",
            "medium": "facebook/musicgen-medium", 
            "large": "facebook/musicgen-large"
        }
        return model_ids.get(size, "facebook/musicgen-medium")
    
    def _load_model(self):
        """Load the MusicGen model and processor."""
        try:
            model_id = self._get_model_id(self.model_size)
            print(f"Loading MusicGen {self.model_size} on {self.device}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Load model with optimized settings
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            print(f"✅ MusicGen {self.model_size} loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        duration: int = 10,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        classifier_free_guidance: float = 3.0
    ) -> tuple:
        """
        Generate audio from a text prompt.
        
        Args:
            prompt (str): Text description of the audio/music to generate
            duration (int): Duration in seconds (5-30)
            temperature (float): Controls randomness (0.1-2.0)
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p sampling parameter
            classifier_free_guidance (float): Guidance scale for better quality
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Please initialize the generator first.")
        
        try:
            # Prepare inputs
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate audio
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=classifier_free_guidance,
                    max_new_tokens=int(duration * 50),  # Approximate tokens per second
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Convert to numpy array
            audio_data = audio_values.cpu().numpy().squeeze()
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Get sample rate (MusicGen uses 32kHz)
            sample_rate = 32000
            
            # Clear GPU memory if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            gc.collect()
            
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"❌ Error generating audio: {str(e)}")
            raise
    
    def generate_with_continuation(
        self,
        prompt: str,
        audio_seed: np.ndarray,
        duration: int = 10,
        temperature: float = 1.0
    ) -> tuple:
        """
        Generate audio that continues from an existing audio seed.
        
        Args:
            prompt (str): Text description for the continuation
            audio_seed (np.ndarray): Existing audio to continue from
            duration (int): Duration of continuation in seconds
            temperature (float): Controls randomness
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Please initialize the generator first.")
        
        try:
            # Prepare inputs
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Convert audio seed to tensor
            audio_seed_tensor = torch.from_numpy(audio_seed).unsqueeze(0).to(self.device)
            
            # Generate continuation
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=int(duration * 50),
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    audio_seed=audio_seed_tensor
                )
            
            # Convert to numpy array
            audio_data = audio_values.cpu().numpy().squeeze()
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            sample_rate = 32000
            
            # Clear GPU memory if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            gc.collect()
            
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"❌ Error generating audio continuation: {str(e)}")
            raise
    
    def get_available_genres(self) -> list:
        """Get a list of suggested genres for better prompts."""
        return [
            "Classical", "Jazz", "Rock", "Pop", "Electronic", "Folk", "Blues",
            "Country", "Hip Hop", "R&B", "Reggae", "Latin", "World", "Ambient",
            "Soundtrack", "Choral", "Orchestral", "Piano", "Guitar", "Violin"
        ]
    
    def get_prompt_suggestions(self) -> list:
        """Get example prompts for inspiration."""
        return [
            "A peaceful piano melody with soft strings in the background",
            "Upbeat electronic dance music with heavy bass",
            "Gentle acoustic guitar with nature sounds",
            "Epic orchestral music for a fantasy adventure",
            "Smooth jazz with saxophone and piano",
            "Energetic rock music with electric guitar",
            "Calming ambient music with synthesizers",
            "Traditional folk music with acoustic instruments",
            "Modern pop music with catchy melodies",
            "Classical symphony with full orchestra"
        ]
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "loaded": self.model is not None,
            "sample_rate": 32000,
            "max_duration": 30  # seconds
        }
    
    def cleanup(self):
        """Clean up resources and free memory."""
        if self.model:
            del self.model
            self.model = None
        
        if self.processor:
            del self.processor
            self.processor = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()
        print("🧹 Audio generator cleaned up")

# Example usage and testing
if __name__ == "__main__":
    # Test the audio generator
    generator = LocalAudioGenerator(model_size="small")  # Use small for testing
    
    # Generate a test audio
    test_prompt = "A peaceful piano melody with soft strings in the background"
    audio_data, sample_rate = generator.generate(
        prompt=test_prompt,
        duration=5,  # Short duration for testing
        temperature=1.0
    )
    
    # Save the test audio
    import soundfile as sf
    sf.write("test_generated_audio.wav", audio_data, sample_rate)
    print("✅ Test audio generated and saved as 'test_generated_audio.wav'")
    
    # Cleanup
    generator.cleanup() 