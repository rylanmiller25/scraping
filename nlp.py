import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional
import logging
import numpy as np

# Set up logger
logger = logging.getLogger("startup_scraper")

class NLPEngine:
    def __init__(self, model_name: str = 'jinaai/jina-embeddings-v2-base-en'):
        """
        Initializes the NLP engine with the Jina embeddings model.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading NLP model {model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.model.eval() # Set to evaluation mode
            logger.info("NLP model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NLP model: {e}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates a vector embedding for the given text.
        Handles tokenization and truncation to 8192 tokens.
        """
        if not text:
            # Return zero vector or handle as empty? 
            # Usually empty text implies some failure handled upstream, 
            # but for robustness return a zero vector of correct dim (768 for base-en).
            # Jina v2 base is 768 dimensions.
            return np.zeros(768)

        # Tokenize and truncate
        # The model supports 8192 context length.
        # We perform deterministic truncation to the first 8192 tokens.
        inputs = self.tokenizer(
            text, 
            max_length=8192, 
            truncation=True, 
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Jina v2 uses mean pooling for sentence embeddings usually, 
            # or the model output might have a specific pooler. 
            # The huggingface model card says: "use the mean pooling of the last hidden state"
            # However, AutoModel usually returns last_hidden_state. 
            # Let's use mean pooling.
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean Pooling - Take attention mask into account for correct averaging
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # Convert to numpy
            return embeddings.cpu().numpy().flatten()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Computes the cosine similarity between two texts.
        Returns a float between -1.0 and 1.0.
        """
        # Short circuit for exact string match
        if text1 == text2:
            return 1.0
        
        vec1 = self.get_embedding(text1)
        vec2 = self.get_embedding(text2)
        
        # Compute Cosine Similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

