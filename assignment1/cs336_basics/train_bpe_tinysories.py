import json
import time
from cs336_basics.train import train_bpe

intput_path = "/home/wangqinshi/data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]

start_time = time.time()
print("start training on TinyStories dataset")
vocab,merges = train_bpe(intput_path,vocab_size,special_tokens)
end_time = time.time()
print("training finished")
print(f"Total time used: {end_time - start_time} seconds")

with open("tinystories_vocab.json","w",encoding = "utf-8") as f:
    json.dump({k.decode('utf-8') :v.decode('utf-8') for k,v in vocab.items()},f,ensure_ascii=False,indent=4)
with open("tinystories_merges.txt","w",encoding = "utf-8") as f:
    for merge in merges:
        f.write(merge[0].decode('utf-8') + " " + merge[1].decode('utf-8') + "\n")
        