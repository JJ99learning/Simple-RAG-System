import requests
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

class JinaEmbedder:
    def __init__(self, api_key: str = None):
        """Initialize the Jina embedder with API key.
        
        Args:
            api_key (str, optional): Jina API key. If not provided, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv('JINA_API_KEY')
        if not self.api_key:
            raise ValueError("Jina API key is required. Either pass it to the constructor or set JINA_API_KEY environment variable.")
            
        self.url = 'https://api.jina.ai/v1/embeddings'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        self.model = "jina-clip-v2"
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string.
        
        Args:
            text (str): The text to embed
            
        Returns:
            List[float]: The embedding vector
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        data = {
            "model": self.model,
            "input": [{"text": text} for text in texts]
        }
        
        response = requests.post(self.url, json=data, headers=self.headers)
        response.raise_for_status()  # Raise exception for bad status codes
        
        result = response.json()
        return [item['embedding'] for item in result['data']]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents for LangChain compatibility.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.embed_texts(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string for LangChain compatibility.
        
        Args:
            text (str): The query text to embed
            
        Returns:
            List[float]: The embedding vector
        """
        return self.embed_text(text)
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Make the class callable to match Chroma's embedding function interface.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.embed_texts(texts) 