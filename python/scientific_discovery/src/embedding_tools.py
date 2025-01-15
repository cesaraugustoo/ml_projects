from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

class EmbeddingTools:
    """
    A class to manage embedding model loading and generation.
    """
   
    def __init__(self, model_name="bert-base-uncased"):
        """Initialize the embedding model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_batch_embeddings(self, texts, batch_size=32):
        """Generate embeddings for a batch of texts."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)

    def compare_embeddings(self, text1, text2):
        """Compare embeddings of two texts using cosine similarity."""
        emb1 = self.generate_batch_embeddings(text1).flatten()
        emb2 = self.generate_batch_embeddings(text2).flatten()
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return similarity

    @staticmethod
    def load_custom_model(model_path):
        """Load a custom model from a local path."""
        return AutoModel.from_pretrained(model_path)

    @staticmethod
    def load_custom_tokenizer(model_path):
        """Load a custom tokenizer from a local path."""
        return AutoTokenizer.from_pretrained(model_path)