import torch
import sys


from bbb_model import SPMM_classifier
from transformers import BertTokenizer, WordpieceTokenizer

class BBBPredictor:
    """Lightweight wrapper for BBB prediction"""
    
    def __init__(self, model_path='spmm_classifier.pth', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer(
            vocab_file=checkpoint['tokenizer_vocab'], 
            do_lower_case=False, 
            do_basic_tokenize=False
        )
        self.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.tokenizer.vocab, 
            unk_token=self.tokenizer.unk_token, 
            max_input_chars_per_word=250
        )
        
        # Initialize and load model
        self.model = SPMM_classifier(config=checkpoint['config'], tokenizer=self.tokenizer)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, smiles):
        """Predict single SMILES"""
        return self.model.predict(smiles, device=self.device)
    
    def predict_batch(self, smiles_list, batch_size=32):
        """Predict multiple SMILES"""
        return self.model.predict_batch(smiles_list, device=self.device, batch_size=batch_size)
    
    def __call__(self, smiles):
        """Allow predictor(smiles) syntax"""
        if isinstance(smiles, list):
            return self.predict_batch(smiles)
        return self.predict(smiles)



# if __name__ == "__main__":
#     # Initialize once
#     predictor = BBBPredictor('spmm_classifier.pth')
    
#     # Single prediction
#     result = predictor("c1ccccc1")
#     print(result['bbb_permeable'], result['confidence'])
    
#     # Or use as callable
#     result = predictor("CC(=O)Oc1ccccc1C(=O)O")
    
#     # Batch prediction
#     results = predictor(["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])