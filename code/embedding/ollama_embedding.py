# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Ollama embedding implementation.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

import os
import asyncio
import threading
from typing import List, Optional

from ollama import AsyncClient
from config.config import CONFIG

from utils.logging_config_helper import get_configured_logger, LogLevel
logger = get_configured_logger("ollama_embedding")

# Add lock for thread-safe client access
_client_lock = threading.Lock()
ollama_client = None

def get_async_client() -> AsyncClient:
    """
    Configure and return an asynchronous Ollama client.
    """
    global ollama_client
    with _client_lock:  # Thread-safe client initialization
        if ollama_client is None:
            try:
                ollama_client = AsyncClient()
                logger.debug("Ollama client initialized successfully")
            except Exception as e:
                logger.exception("Failed to initialize Ollama client")
                raise
    
    return ollama_client

async def get_ollama_embeddings(
    text: str,
    model: Optional[str] = None,
    timeout: float = 30.0
) -> List[float]:
    """
    Generate an embedding for a single text using Ollama API.
    
    Args:
        text: The text to embed
        model: Optional model ID to use, defaults to provider's configured model
        timeout: Maximum time to wait for the embedding response in seconds
        
    Returns:
        List of floats representing the embedding vector
    """
    # If model not provided, get it from config
    if model is None:
        provider_config = CONFIG.get_embedding_provider("ollama")
        if provider_config and provider_config.model:
            model = provider_config.model
        else:
            # Default to a common embedding model
            model = "nomic-embed-text"
    
    logger.debug(f"Generating Ollama embedding with model: {model}")
    logger.debug(f"Text length: {len(text)} chars")
    
    client = get_async_client()

    try:
        # Clean input text (replace newlines with spaces)
        text = text.replace("\n", " ")
        response = await client.embed(
            input=text,
            model=model,
            keep_alive=timeout
        )
        
        embedding = response.embeddings
        logger.debug(f"Ollama embedding generated, dimension: {len(embedding)}")
        return embedding
    except Exception as e:
        logger.exception("Error generating Ollama embedding")
        logger.log_with_context(
            LogLevel.ERROR,
            "Ollama embedding generation failed",
            {
                "model": model,
                "text_length": len(text),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        raise

async def get_ollama_batch_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    timeout: float = 60.0
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts using Ollama API.
    
    Args:
        texts: List of texts to embed
        model: Optional model ID to use, defaults to provider's configured model
        timeout: Maximum time to wait for the batch embedding response in seconds
        
    Returns:
        List of embedding vectors, each a list of floats
    """
    # If model not provided, get it from config
    if model is None:
        provider_config = CONFIG.get_embedding_provider("ollama")
        if provider_config and provider_config.model:
            model = provider_config.model
        else:
            # Default to a common embedding model
            model = "nomic-embed-text"
    
    logger.debug(f"Generating Ollama batch embeddings with model: {model}")
    logger.debug(f"Batch size: {len(texts)} texts")
    
    client = get_async_client()

    try:
        # Clean input texts (replace newlines with spaces)
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        response = await client.embed(
            input=cleaned_texts,
            model=model,
            keep_alive=timeout
        )
        
        embeddings = response.embeddings
        logger.debug(f"Ollama batch embeddings generated, count: {len(embeddings)}")
        return embeddings
    except Exception as e:
        logger.exception("Error generating Ollama batch embeddings")
        logger.log_with_context(
            LogLevel.ERROR,
            "Ollama batch embedding generation failed",
            {
                "model": model,
                "batch_size": len(texts),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        raise