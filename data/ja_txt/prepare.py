import os
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'ja.txt')

# 指定ファイルのサイズを取得
file_size = os.path.getsize(input_file_path)
print(f"ja.txt has {file_size:,} bytes")

# 分割するときの最大のファイルサイズ
max_file_size = 1 * 1024 * 1024 * 1024 # 1GB

with open(input_file_path, 'r', encoding='utf-8') as f:
    file_count = 0
    while True:
        # ファイルサイズがmax_file_sizeを超えるまで読み込む
        data = f.read(max_file_size)
        n = len(data)
        print(f"read {n:,} bytes")

        # 保存するファイル名を変更する
        save_txt_file_name = f'input_{n}.txt'
        print(f"save {save_txt_file_name}")

        # ファイルに書き込む
        save_file_name = f'input_{file_count}.txt'
        with open(save_file_name, 'w', encoding='utf-8') as f2:
            f2.write(data)       
            
        file_count += 1 
        break
    
exit()
        
n = len(load_text)
train_data = load_text[:int(n*0.9)]
val_data = load_text[int(n*0.9):]

print(f'train has {len(train_data):,} characters')
print(f'val has {len(val_data):,} characters')

# 学習データを一つ表示する
print(train_data[:100])

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


# 保存したデータのファイルサイズを表示
print(f"train.bin has {os.path.getsize(os.path.join(os.path.dirname(__file__), 'train.bin')):,} bytes")
print(f"val.bin has {os.path.getsize(os.path.join(os.path.dirname(__file__), 'val.bin')):,} bytes")