
import regex as re

GPT4_REGEX_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def get_pair_stats(ids : list, counts:dict=None) -> dict:
    """Gets counts of consecutive pairs in a list of integers

    Args:
        ids (list): list of integers
        counts (dict, optional): exisiting dictionary of counts to be updated. Defaults to None.

    Returns:
        dict: updated dictionary of counts of consecutive pairs
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_pairs(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids
    
    
    pass

class Tokenizer:
    def __init__(self, regex_pattern=None):
        """
        Tokenizer class
        Splits inputs based on regex pattern

        Args:
            regex_pattern (_type_, optional): _description_. Defaults to None.
        """
        self.regex_pattern = GPT4_REGEX_PATTERN if regex_pattern is None else regex_pattern
        self.compiled_regex_pattern = re.compile(self.regex_pattern)
    
    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256
        
        text_pieces = re.findall(self.compiled_regex_pattern, text)
        
        # encode text pieces into utf-8
        ids = [list(piece.encode("utf-8")) for piece in text_pieces]
        
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            pair_stats = {}
            
            for piece_ids in ids:
                get_pair_stats(piece_ids, pair_stats)
                
            max_pair = max(pair_stats, key=pair_stats.get)
            
            # new token in next id
            idx = 256 + i
            
            # merge all max pairs in all sequences
            ids = [merge_pairs(piece_ids, max_pair, idx) for piece_ids in ids]

            merges[max_pair] = idx
            vocab[idx] = vocab[max_pair[0]] + vocab[max_pair[1]]

        self.merges = merges
        self.vocab = vocab
    
    def encode_piece(self, text_bytes):
        ids = list(text_bytes)
        
        while len(ids) >= 2:
            pair_stats = get_pair_stats(ids)
            
            pair = min(pair_stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                # All done, no more pairs
                break
        
            idx = self.merges[pair]
            ids = merge_pairs(ids, pair, idx)
        return ids
    
    def encode(self, text):
        text_pieces = re.findall(self.compiled_regex_pattern, text)
        
        ids = []
        for piece in text_pieces:
            pieces_bytes = piece.encode("utf-8")
            piece_ids = self.encode_piece(pieces_bytes)
            ids.extend(piece_ids)
        return ids
    
    def decode(self, ids):
        tokens = b"".join([self.vocab[idx] for idx in ids])
        text = tokens.decode("utf-8", errors="replace")
        return text
