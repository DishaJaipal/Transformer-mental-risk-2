
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def extract_deproberta_embeddings():
    """
    Extract embeddings from DepRoBERTa base model
    """
    
    # Load data
    print("Loading data...")
    combined_df = pd.read_csv('../data/combined_cleaned.csv')
    
    texts = combined_df['text'].tolist()
    labels = combined_df['labels'].values
    
    print(f"Total samples: {len(texts)}")
    
    # Initialize DepRoBERTa base model (for embeddings)
    print("Loading DepRoBERTa base model...")
    model_name = "rafalposwiata/deproberta-large-v1"  # Base model for embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    def extract_embeddings(texts, batch_size=8):
        """Extract embeddings from DepRoBERTa"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(texts)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save embeddings and labels
    os.makedirs('embeddings', exist_ok=True)
    
    np.save('embeddings/deproberta_embeddings.npy', embeddings)
    np.save('embeddings/labels.npy', labels)
    
    print("✓ Embeddings saved to embeddings/")
    print("✓ Ready for RF-LR training!")
    
    return embeddings, labels

if __name__ == '__main__':
    extract_deproberta_embeddings()