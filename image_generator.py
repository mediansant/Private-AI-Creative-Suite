import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import gc

class LocalImageGenerator:
    """
    Local image generation using Stable Diffusion XL.
    Runs entirely on your device for complete privacy.
    """
    
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Initialize the image generator with Stable Diffusion XL.
        
        Args:
            model_id (str): Hugging Face model ID for Stable Diffusion XL
        """
        self.model_id = model_id
        self.pipeline = None
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
    
    def _load_model(self):
        """Load the Stable Diffusion XL model."""
        try:
            print(f"Loading Stable Diffusion XL on {self.device}...")
            
            # Load the pipeline with optimized settings
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device != "cpu" else None
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Use DPM++ 2M scheduler for faster inference
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Enable memory efficient attention if available
            try:
                if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ Xformers memory optimization enabled")
            except Exception as e:
                print(f"⚠️  Xformers not available (optional optimization): {str(e)}")
                print("   This is normal - the app will work without xformers")
            
            # Enable model CPU offload for memory efficiency
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
            
            print(f"✅ Stable Diffusion XL loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 768,
        height: int = 768,
        seed: int = None
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt (str): Text description of the image to generate
            negative_prompt (str): Text description of what to avoid
            num_inference_steps (int): Number of denoising steps (20-50)
            guidance_scale (float): How closely to follow the prompt (1-20)
            width (int): Image width (512, 768, or 1024)
            height (int): Image height (512, 768, or 1024)
            seed (int): Random seed for reproducible results
            
        Returns:
            PIL.Image: Generated image
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Please initialize the generator first.")
        
        # Set default negative prompt if none provided
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        try:
            # Generate the image
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
            )
            
            # Get the generated image
            image = result.images[0]
            
            # Clear GPU memory if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            return image
            
        except Exception as e:
            print(f"❌ Error generating image: {str(e)}")
            raise
    
    def generate_variations(
        self,
        base_image: Image.Image,
        prompt: str,
        strength: float = 0.8,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Generate variations of an existing image.
        
        Args:
            base_image (PIL.Image): Base image to create variations from
            prompt (str): Text description for the variation
            strength (float): How much to change the image (0-1)
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            
        Returns:
            PIL.Image: Generated image variation
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Please initialize the generator first.")
        
        try:
            # Use img2img pipeline for variations
            result = self.pipeline(
                prompt=prompt,
                image=base_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            image = result.images[0]
            
            # Clear GPU memory if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            return image
            
        except Exception as e:
            print(f"❌ Error generating image variation: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "loaded": self.pipeline is not None,
            "dtype": str(self.pipeline.dtype) if self.pipeline else None
        }
    
    def cleanup(self):
        """Clean up resources and free memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()
        print("🧹 Image generator cleaned up")

# Example usage and testing
if __name__ == "__main__":
    # Test the image generator
    generator = LocalImageGenerator()
    
    # Generate a test image
    test_prompt = "A beautiful sunset over mountains, digital art style"
    image = generator.generate(
        prompt=test_prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        width=768,
        height=768
    )
    
    # Save the test image
    image.save("test_generated_image.png")
    print("✅ Test image generated and saved as 'test_generated_image.png'")
    
    # Cleanup
    generator.cleanup() 