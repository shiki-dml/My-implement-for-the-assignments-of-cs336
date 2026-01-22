#BPE tokenizer pattern
import regex as re
import multiprocessing
from collections import Counter
import os
from cs336_basics.pretokenization_example import find_chunk_boundaries
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def word_counting(text,vocab_count,pair_count):
    #used to deal with each part returned by find_chunk_boundaries, in specific,
    #write down the newest token 
    tokens = re.findall(PAT,text)
    for token in tokens:
        word_tuple = tuple(bytes([l]) for l in token.encode('utf-8'))
        pair_tuple = tuple((word_tuple[l],word_tuple[l+1]) for l in range(len(word_tuple)-1))
        if word_tuple in vocab_count:
            vocab_count[word_tuple] += 1
        else:
            vocab_count[word_tuple] = 1
        for pair in pair_tuple:
            if pair in pair_count.keys():
                pair_count[pair]+=1
            else:
                pair_count[pair]=1
            
def update_word_counting_dict(original_tuple,merge_pair):#used to update the word_counting_dict
    i =0
    merge_tuple = ()
    while i < len(original_tuple):
        if   i < len(original_tuple)-1 and (original_tuple[i],original_tuple[i+1]) == merge_pair:
            merge_tuple += (merge_pair[0]+merge_pair[1],)
            i +=2
        else:
            merge_tuple += (original_tuple[i],)
            i +=1

    return merge_tuple 

def _process_chunk_worker(input_path, start, end, main_token_str):
    
    local_word_counts = Counter()
    local_pair_counts = Counter()
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
        
        parts = chunk_text.split(main_token_str)
        
        for part in parts:
            if part: 
                word_counting(part, local_word_counts, local_pair_counts)
                
    return local_word_counts, local_pair_counts

def train_bpe(input_path,vocab_size,special_tokens):
    #input_path:str vocab_size:int special_tokens:list[str]

    merges = [] #initialize merges list
    word_counting_dict ={} #initialize temporary list for each word
    pair_counting_dict = {} #initialize temporary list for each pair
    vocab = {i:bytes([i]) for i in range(256)} #initialize vocab with 256 bytes

    #initialize special_tokens
    idx = 256
    for token in special_tokens:
        vocab[idx] = token.encode('utf-8')
        idx +=1

    main_token = special_tokens[0] #use the first special token as the separator
    sep_bytes = main_token.encode('utf-8')
    num_process = os.cpu_count() #the number of parallel processinges

    num_workers = max(1,multiprocessing.cpu_count()-1)

    with open(input_path,"rb") as f:
        boundaries = find_chunk_boundaries(f,num_workers,sep_bytes)
    tasks = []
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        tasks.append((input_path,start,end,main_token))

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(_process_chunk_worker, tasks)

    for local_words, local_pairs in results:
        for word,count in local_words.items():
            word_counting_dict[word] = word_counting_dict.get(word,0) + count
        for pair,count in local_pairs.items():
            pair_counting_dict[pair] = pair_counting_dict.get(pair,0) + count

    #at this point, word_counting_dict and pair_counting_dict contains all pre-tokenized words and their counts
    #what is next to do for word_counting_dict is to update it according to merges
    while len(vocab) < vocab_size:
        if not pair_counting_dict:
            break
        # find the most frequent pair
        max_pair = max(pair_counting_dict.items(), key=lambda item: (item[1], item[0]))[0]
        
        # store the max_pair into merges
        merges.append(max_pair)
        new_token = max_pair[0] + max_pair[1]
        vocab[idx] = new_token
        idx += 1
        # store the change of words
        changes = []

        current_words = list(word_counting_dict.keys())
        for word in current_words:
            #accelerate
            if max_pair[0] not in word:
                continue
            
            new_word = update_word_counting_dict(word, max_pair)
            
            if new_word != word:
                count = word_counting_dict[word]
                changes.append((word, new_word, count))

        for old_word, new_word, count in changes:
            
            #update word_counting_dict
            del word_counting_dict[old_word]
            word_counting_dict[new_word] = word_counting_dict.get(new_word, 0) + count

            #then updatee pair_counting_dict
            #note that we only need to update the pairs that are affected by the change from old_word to new_word       
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i+1])
                pair_counting_dict[p] -= count
                if pair_counting_dict[p] == 0:
                    del pair_counting_dict[p]
            #append new pairs into pair_counting_dict
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i+1])
                pair_counting_dict[p] = pair_counting_dict.get(p, 0) + count

    return vocab,merges
#vocab:dict[int.bytes] merges:list[tuple[bytes,bytes]]
