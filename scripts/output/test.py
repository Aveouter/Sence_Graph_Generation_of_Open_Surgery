# print npy 文件
import numpy as np
file_path = r"D:\Code\Hsg_gen\scripts\output\text_rel.npy"
embeddings = np.load(file_path)

print("embeddings shape:", embeddings.shape)

for i, emb in enumerate(embeddings):
    print(f"Embedding {i}: {emb[:5]}...")  # 打印前5个元素

