#Tokenizer class
import regex as re
import multiprocessing
from cs336_basics.pretokenization_example import find_chunk_boundaries
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self,vocab,merges,special_tokens = None):
        #vocab:dict[int:bytes] merges:list[tuple[bytes,bytes]]
        #special_tokens:list[str] | None = None
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        #vocab_bytes_to_id: dict[bytes:int]
        self.vocab_bytes_to_id = {v:k for k,v in vocab.items()}

        #merge_ranks: dict[tuple[bytes,bytes]:int]
        self.merge_ranks = {pair:idx for idx,pair in enumerate(merges)}

        #special_tokens_ids: dict[str:int]
        self.special_tokens_ids = {}
        if self.special_tokens:
            for st in self.special_tokens:
                st_bytes = st.encode('utf-8')
                if st_bytes in self.vocab_bytes_to_id:
                    self.special_tokens_ids[st] = self.vocab_bytes_to_id[st_bytes]

        self.pat_complied = re.compile(PAT)
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens = None):
        #this class method loads vocab and merges from files and return a Tokenizer instance
        #vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
        vocab = {}
        with open(vocab_filepath,'rb') as vf:
            for line in vf:
                line_content = line.rstrip(b'\n')
                if not line_content:
                    continue  #skip empty lines
                parts = line_content.split(b'\t',1)
                if len(parts) != 2:
                    continue  #skip malformed lines
                idx,token = parts
                vocab[int(idx)] = token
        merges = []
        with open(merges_filepath,'rb') as mf:
            for line in mf:
                line_content = line.rstrip(b'\n')
                if not line_content:
                    continue 
                parts = line_content.split(b'\t',1)
                if len(parts) != 2:
                    continue 
                first,second = parts
                merges.append((first,second))
        return cls(vocab,merges,special_tokens)
    
    def _merge_tokens(self,tokens:list[bytes]) -> list[bytes]:
        while len(tokens) >1:
            min_rank = float('inf')
            min_pair = None
            for i in range(len(tokens)-1):
                pair = (tokens[i],tokens[i+1])
                rank = self.merge_ranks.get(pair,float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None:
                break
            new_tokens = []
            i =0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i],tokens[i+1]) == min_pair:
                    new_tokens.append(tokens[i]+tokens[i+1])
                    i +=2
                else:
                    new_tokens.append(tokens[i])
                    i +=1
            tokens = new_tokens
        return tokens

    def encode(self,text:str) -> list[int]:
        #encode an input text into a sequence of token IDs
        ids = []
        #first split the special tokens
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens,key=len,reverse=True)
            special_pattren = "(" + "|".join(re.escape(st) for st in sorted_special_tokens) + ")"
            parts = re.split(special_pattren,text)
        else:
            parts = [text]
        
        for part in parts:
            if not part: continue
            #if part is a special token
            if part in self.special_tokens_ids:
                ids.append(self.special_tokens_ids[part])
                continue
            #if part in normal text, we run BPE encoding
            text_tokens = re.findall(self.pat_complied,part)
            for token_str in text_tokens:
                token_bytes = [bytes([b]) for b in token_str.encode('utf-8')]
                merged_tokens = self._merge_tokens(token_bytes)

                for mt in merged_tokens:
                    if mt in self.vocab_bytes_to_id:
                        ids.append(self.vocab_bytes_to_id[mt])
        return ids
    def encode_iterable(self, iterable):
        #encode an iterable of strings into an iterator of token IDs
        for text_ in iterable:
            text_ids = self.encode(text_)
            yield from text_ids
    def decode(self, ids: list[int]) -> str:
        #decode a sequence of token IDs back into a string
        bytes_list = []
        for id in ids:
            if id in self.vocab:
                bytes_list.append(self.vocab[id])
        decoded_bytes = b''.join(bytes_list)
        return decoded_bytes.decode('utf-8',errors='replace')