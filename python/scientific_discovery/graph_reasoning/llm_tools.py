from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any
from pathlib import Path
import logging
import base64
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BaseAIConfig:
    """Base configuration for AI API calls."""
    api_key: str
    max_tokens: int = 2048
    temperature: float = 0.2
    timeout: int = 120

@dataclass
class OpenAIConfig(BaseAIConfig):
    """Configuration for OpenAI API calls."""
    organization: str = ""
    gpt_model: str = "gpt-4-0125-preview"
    dalle_model: str = "dall-e-3"
    frequency_penalty: float = 0
    presence_penalty: float = 0
    top_p: float = 1.0

@dataclass
class GeminiConfig(BaseAIConfig):
    """Configuration for Google's Gemini API calls."""
    model_name: str = "gemini-pro"
    vision_model: str = "gemini-pro-vision"
    image_model: str = "gemini-1.0-pro-vision"
    top_k: int = 40
    top_p: float = 0.8

class AIClient(ABC):
    """Abstract base class for AI clients."""
    
    @abstractmethod
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Generate text using AI model."""
        pass

    @abstractmethod
    def analyze_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Union[str, Path],
        **kwargs
    ) -> str:
        """Analyze an image using AI model."""
        pass

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        output_dir: Union[str, Path],
        **kwargs
    ) -> List[Path]:
        """Generate images using AI model."""
        pass

class OpenAIClient(AIClient):
    """Client for interacting with OpenAI's APIs."""
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            organization=config.organization
        )

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            completion = self.client.chat.completions.create(
                model=self.config.gpt_model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
                frequency_penalty=kwargs.get('frequency_penalty', self.config.frequency_penalty),
                presence_penalty=kwargs.get('presence_penalty', self.config.presence_penalty)
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise

    def analyze_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Union[str, Path],
        **kwargs
    ) -> str:
        try:
            image_base64 = self._encode_image(image_path)
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error analyzing image with OpenAI: {str(e)}")
            raise

    def generate_image(
        self,
        prompt: str,
        output_dir: Union[str, Path],
        **kwargs
    ) -> List[Path]:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            response = self.client.images.generate(
                model=self.config.dalle_model,
                prompt=prompt,
                n=kwargs.get('n', 1),
                size=kwargs.get('size', "1024x1024"),
                quality=kwargs.get('quality', "standard"),
                response_format="b64_json"
            )
            
            generated_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for idx, image_data in enumerate(response.data):
                image_bytes = base64.b64decode(image_data.b64_json)
                filename = output_dir / f"generated_image_{timestamp}_{idx}.png"
                
                with open(filename, "wb") as f:
                    f.write(image_bytes)
                generated_files.append(filename)
            
            return generated_files
        except Exception as e:
            logger.error(f"Error generating image with OpenAI: {str(e)}")
            raise

    @staticmethod
    def _encode_image(image_path: Union[str, Path]) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

class GeminiClient(AIClient):
    """Client for interacting with Google's Gemini APIs."""
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        genai.configure(api_key=config.api_key)
        # Initialize models with system instruction capability
        self.model = genai.GenerativeModel(
            config.model_name,
            system_instruction=None  # Will be set per request
        )
        self.vision_model = genai.GenerativeModel(
            config.vision_model,
            system_instruction=None  # Will be set per request
        )

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        try:
            # Create a new model instance with the system prompt
            model = genai.GenerativeModel(
                self.config.model_name,
                system_instruction=system_prompt
            )
            
            response = model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', self.config.temperature),
                    top_p=kwargs.get('top_p', self.config.top_p),
                    top_k=kwargs.get('top_k', self.config.top_k),
                    max_output_tokens=kwargs.get('max_tokens', self.config.max_tokens)
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {str(e)}")
            raise

    def analyze_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Union[str, Path],
        **kwargs
    ) -> str:
        try:
            # Create a new vision model instance with the system prompt
            vision_model = genai.GenerativeModel(
                self.config.vision_model,
                system_instruction=system_prompt
            )
            
            image = Image.open(image_path)
            
            response = vision_model.generate_content(
                [user_prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', self.config.temperature),
                    top_p=kwargs.get('top_p', self.config.top_p),
                    top_k=kwargs.get('top_k', self.config.top_k),
                    max_output_tokens=kwargs.get('max_tokens', self.config.max_tokens)
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {str(e)}")
            raise

    def generate_image(
        self,
        prompt: str,
        output_dir: Union[str, Path],
        **kwargs
    ) -> List[Path]:
        raise NotImplementedError("Gemini does not support image generation yet")

class ChatSession:
    """Manage a chat session with context for any AI client."""
    
    def __init__(self, client: AIClient):
        self.client = client
        self.messages: List[Dict[str, Any]] = []
        self.system_message: Optional[str] = None

    def set_system_message(self, message: str) -> None:
        """Set the system message for the chat session."""
        self.system_message = message
        self.messages = [{"role": "system", "content": message}]

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        self.messages.append({"role": role, "content": content})

    def get_response(
        self,
        query: str,
        image_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """Get a response from the model, optionally including an image."""
        try:
            if not self.system_message:
                self.set_system_message("You are a helpful AI assistant.")

            if image_path:
                response = self.client.analyze_image(
                    system_prompt=self.system_message,
                    user_prompt=query,
                    image_path=image_path,
                    **kwargs
                )
            else:
                response = self.client.generate_text(
                    system_prompt=self.system_message,
                    user_prompt=query,
                    **kwargs
                )
            
            self.add_message("user", query)
            self.add_message("assistant", response)
            return response
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise