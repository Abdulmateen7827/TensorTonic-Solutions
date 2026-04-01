import numpy as np
from typing import List, Dict
from collections import Counter
class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        all_texts = [t.split(" ") for t in texts]
        count = Counter()
        for all in all_texts:
            count.update(all)
        self.word_to_id.update({self.pad_token: 0, self.unk_token: 1,
                                self.bos_token: 2, self.eos_token: 3})
        for idx, word in enumerate(count.keys(), start=4):
            self.word_to_id[word] = idx
        
        self.vocab_size = len(self.word_to_id)
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        texts = text.lower().split()

        token_ids = [
            self.word_to_id.get(i, self.word_to_id[self.unk_token])
            for i in texts
        ]
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        text = ""

        return " ".join(self.id_to_word.get(i, self.unk_token) for i in ids)
        
            
        

