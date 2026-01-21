from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.train import train_bpe
text = "hello world math stack exchange computer science artificial intelligence <|endoftext|>"
vocab,merge = train_bpe(text,50,["<|endoftext|>"])
print(vocab)
print(merge)