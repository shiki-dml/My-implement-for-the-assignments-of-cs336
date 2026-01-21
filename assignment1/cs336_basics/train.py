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
        max_pair = max(pair_counting_dict.items(),key = lambda item :(item[1],item[0]))[0]
        #note that item[1] is the count, item[0] is the pair itself, so we first sory by count
        #then sory in lexicographical order 

        merges.append(max_pair)
        vocab[idx] = max_pair[0]+max_pair[1] #the max_pair is a tuple of two elements
        idx +=1

        test_tuple = (max_pair[0],max_pair[1])
        reset_tuple = max_pair[0]+max_pair[1] 
        #used for testing if the max_pair exists in the tuple of word_counting_dict
        #reset_tuple is used to update the pair_counting_dict
            
        #then we update the word_couting_dict and pair_counting_dict
        key_to_delete_word = []
        key_to_add_word = []
        count_to_add_word = []
        
        for tup in word_counting_dict:
            num_1 = len(tup)
            if num_1 <2:
                continue
            else:
                for i in range(num_1-1):
                    if tup[i] == max_pair[0] and tup[i+1] == max_pair[1]:
                        new_tuple = update_word_counting_dict(tup,test_tuple)
                        count = word_counting_dict[tup]
                        key_to_add_word.append(new_tuple)
                        count_to_add_word.append(count)
                        key_to_delete_word.append(tup)
                        break
                    #test_tuple in tup, note that this process updates each element
                    #of word_counting_dict which contains the max_pair
            

        for k in key_to_delete_word:
            del word_counting_dict[k]
        for k,c in zip(key_to_add_word,count_to_add_word):
            if k in word_counting_dict:
                word_counting_dict[k] += c
            else:
                word_counting_dict[k] = c

        #to update pair_counting_dict, we re-run it on word_counting_dict
        pair_counting_dict = {}
        for token in word_counting_dict:
            count = word_counting_dict[token]
            length = len(token)
            if length <2:
                continue
            else:
                for i in range(length-1):
                    pair_tuple = (token[i],token[i+1])
                    if pair_tuple in pair_counting_dict:
                        pair_counting_dict[pair_tuple] += count
                    else:
                        pair_counting_dict[pair_tuple] = count

    return vocab,merges
#vocab:dict[int.bytes] merges:list[tuple[bytes,bytes]]
