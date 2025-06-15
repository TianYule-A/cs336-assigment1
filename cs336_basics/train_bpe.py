import regex as re
import os
from collections import Counter
import os.path as osp
from cs336_basics.pretokenization_example import find_chunk_boundaries

def train_bpe(
        input_path:str,
        vocab_size:int,
        special_tokens:list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the input text file.

    Args:
        input_path (str): Path to the input text file.
        vocab_size (int): 最终词表的最大值
        special_tokens (list[str]): 特殊token列表

    Returns:
        vocab (dict[int, bytes]):词表中token ID到token字节的映射。
        merges (list[tuple[bytes, bytes]]): 合并字节对列表
    """
    assert vocab_size > 256, "Vocabulary size must be greater than 256"
    # 1.初始化词表
    vocab:dict[int,bytes] = {i:bytes([i]) for i in range(256)}
    merges:list[tuple[bytes, bytes]] = []
    next_token_id = 256  
    existing_bytes_token:set[bytes] = set(vocab.values())# 集合记录词表中已合并出的字节token和特殊token，用于高效索引
    # 添加special tokens到词表
    for sp_token in special_tokens:
        if len(vocab) >= vocab_size:
            break
        sp_token_bytes = sp_token.encode('utf-8')
        if sp_token_bytes not in existing_bytes_token:
            vocab[next_token_id] = sp_token_bytes
            existing_bytes_token.add(sp_token_bytes)
            next_token_id += 1

    # 2.pre-tokenization
    pre_token_counter:Counter[tuple[bytes,...]] = pre_tokenization(input_path, special_tokens)
    # 3.迭代合并字节对
    while len(vocab) < vocab_size:
        # 找到频率最高的字节对
        most_common_pair = get_stat(pre_token_counter,existing_bytes_token)
        if not most_common_pair:
            break
        b0, b1 = most_common_pair
        merge_bytes = b0 + b1

        # 将合并后的token添加到词表
        vocab[next_token_id] = merge_bytes
        existing_bytes_token.add(merge_bytes)
        merges.append((b0, b1))
        next_token_id += 1
        # 更新pre-token计数器
        pre_token_counter = merge(pre_token_counter, b0, b1)

    return vocab, merges

def pre_tokenization(
        input_path:str,
        special_tokens:list[str],
        num_processes:int = 8
) -> Counter[tuple[bytes,...]]:
    """
    对输入文本进行预分词，统计每个pre-token的出现次数。
    Args:
        input_path (str): 输入文本文件路径。
        special_tokens (list[str]): 特殊token列表。
        num_processes (int): 并行处理的进程数。| 切分文本的chunk数
    Returns:
        Counter[bytes]: 预分词后的字节token及其出现次数。
    """
    split_pattern:str = "|".join([re.escape(special_token) for special_token in set(special_tokens)])
    # 预分词正则
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_token_counter:Counter[tuple[bytes,...]] = Counter() # 预分词频率表

    with open(input_path, 'rb') as f:
        boundaries:list[int] = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        # 切分文件为多个部分
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            # 以字节形式读取，解码为字符串
            chunk:str = f.read(end - start).decode("utf-8", errors="ignore")
            # 删除特殊token
            # 将chunk按特殊token进行分割
            if not split_pattern:
                chunk_no_special_token:list[str] = [chunk]
            else:
                chunk_no_special_token:list[str] = re.split(split_pattern, chunk)
            # 对每个chunk进行预分词
            for sub_chunk in chunk_no_special_token:
                if not sub_chunk or sub_chunk.isspace():
                    continue
                for match in re.finditer(PAT, sub_chunk):
                    pre_token = match.group(0).encode('utf-8')
                    pre_token = tuple(bytes([b]) for b in pre_token) 
                    pre_token_counter[pre_token] += 1
            
    return pre_token_counter
    
def get_stat(
        pre_token_counter:Counter[tuple[bytes,...]],
        existing_bytes_token:set[bytes], 
) -> tuple[bytes, bytes]:
    """
    在预分词token上进行字节对编码，获取出现频率最高的字节对。
    
    Args:
        pre_token_counter (Counter[bytes]): 预分词后的字节token及其出现次数。
    
    Returns:
        tuple[bytes, bytes]: 出现频率最高的字节对。
    """
    # 统计所有字节对的频率
    pair_counter:Counter[tuple[bytes,bytes]] = Counter()
    for pre_token, count in pre_token_counter.items():
        # 如果pre_token已经在词表中，跳过
        if len(pre_token)==1 and pre_token[0] in existing_bytes_token:
            continue
        for b0,b1 in zip(pre_token[:-1], pre_token[1:]):
            pair_counter[(b0, b1)] += count
    # 获取出现频率最高的字节对
    most_common_pair = pair_counter.most_common(1)[0][0]
    return most_common_pair

def merge(
        pre_token_counter:Counter[tuple[bytes,...]],
        b0:bytes,
        b1:bytes,
) -> Counter[tuple[bytes,...]]:
    """
    在预分词计数器中合并字节对。

    Args:
        pre_token_counter (Counter[bytes]): 预分词后的字节token及其出现次数。
        b0 (bytes): 第一个字节。
        b1 (bytes): 第二个字节。
        merge_bytes (bytes): 合并后的字节token。

    Returns:
        Counter[bytes]: 更新后的预分词计数器。
    """
    # 更新计数器
    new_counter = Counter()
    merge_bytes = b0 + b1
    for pre_token,count in pre_token_counter.items():
        # pre_token, count = elment, pre_token_counter[elment]
        new_token_list = []
        if len(pre_token) == 1:
            # 如果只有一个token，无法再合并，跳过
            continue
        i = 0 
        while i < len(pre_token) - 1:
            if (pre_token[i] == b0 and pre_token[i + 1] == b1):
                # 如果当前token是要合并的字节对，替换为合并后的token
                new_token_list.append(merge_bytes)
                i += 2
            else:
                # 否则，保留当前token
                new_token_list.append(pre_token[i])
                i += 1
        if i < len(pre_token):
            # 如果还有剩余的token，添加到新token列表
            new_token_list.append(pre_token[i])
        new_token = tuple(new_token_list)
        new_counter[new_token] += count
    return new_counter

if __name__ == "__main__":
    import sys
    input_path:str = osp.join(os.getcwd(),"data/TinyStoriesV2-GPT4-valid.txt") 
    input_path = "/home/tian/cs-learning/python-project/deeplearning/cs336-2025spring/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    print("Input path:", input_path)
    # sys.exit(0) 
    vocab_size:int = 1024
    special_tokens:list[str] = ["<|endoftext|>", "<|startoftext|>", ]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print("Vocabulary size:", len(vocab))
    print("Number of merges:", len(merges))
    print("Sample vocabulary:", list(vocab.items())[:10])
    print("Sample merges:", merges[:100])