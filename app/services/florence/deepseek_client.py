"""
DeepSeek Client for Florence PhD Agent - Fixed version
"""

import os
import json
from typing import Dict, Any, Optional
import requests
from app.core.config import settings


class DeepSeekClient:
    """Client for DeepSeek API with all necessary methods - FIXED"""
    
    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.model = settings.DEEPSEEK_MODEL
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not configured in settings")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"🔧 DeepSeekClient initialized: {self.model}")
    
    def chat_completion(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Send chat completion request to DeepSeek - FIXED VERSION"""
        url = f"{self.base_url}/chat/completions"
        
        # Default parameters - CORREGIDO: usar max_tokens en lugar de otro nombre
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000)  # CORREGIDO
        }
        
        # Add optional parameters
        optional_params = ["stream", "top_p", "frequency_penalty", "presence_penalty"]
        for param in optional_params:
            if param in kwargs:
                data[param] = kwargs[param]
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "content": result["choices"][0]["message"]["content"],
                "model": result["model"],
                "usage": result.get("usage", {}),
                "finish_reason": result["choices"][0]["finish_reason"],
                "id": result["id"]
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek API request failed: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise Exception(f"Invalid API response: {str(e)}")
    
    def research_assistance(self, prompt: str, context: str = "", **kwargs) -> str:
        """Get research assistance from DeepSeek"""
        messages = []
        
        # System message with context
        system_content = """You are Florence, a PhD-level research assistant specializing in statistical analysis and academic research. 
You provide detailed, accurate, and well-referenced responses following academic standards."""
        
        if context:
            system_content = f"Context information:\n{context}\n\n{system_content}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # User prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Get response - CORREGIDO: pasar kwargs correctamente
        response = self.chat_completion(messages, **kwargs)
        return response["content"]
    
    def test_connection(self) -> bool:
        """Test connection to DeepSeek API"""
        try:
            # Simple test request
            messages = [{"role": "user", "content": "Say 'OK'"}]
            response = self.chat_completion(messages, max_tokens=5)
            return "OK" in response["content"].upper()
        except:
            return False
    
    def get_available_models(self) -> list:
        """Get available models from DeepSeek"""
        url = f"{self.base_url}/models"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return [model["id"] for model in result.get("data", [])]
        except:
            return [self.model]  # Return at least the configured model


# Create global instance
deepseek_client = DeepSeekClient()
