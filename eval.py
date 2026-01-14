import faiss
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    output_dir = '/mnt/public/lyy/contrastive_learning_xrd/train_output/peaks_ep50'
    # 加载保存的向量
    cif_emb_dir = '/mnt/minio/battery/xrd/datasets/contrastive_learning/peaks_epoch44/cif'
    embeddings = []
    ids = []
    print('加载中...')
    for filename in os.listdir(cif_emb_dir):
        if filename.endswith('.npy'):
            embedding = np.load(os.path.join(cif_emb_dir, filename))
            embeddings.append(embedding)
            ids.append(int(filename.replace('.npy', '').replace('mp-', '')))
    embeddings = np.array(embeddings)  
    ids = np.array(ids)
    
    print(f"加载了 {embeddings.shape[0]} 个向量，维度 {embeddings.shape[1]}")

    # 创建索引
    dimension = embeddings.shape[1]

    # 使用IVFFlat（适合大数据集）
    quantizer = faiss.IndexFlatL2(dimension)  # 使用L2距离
    nlist = 100  # 聚类中心数
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    # 训练索引
    index.train(embeddings)
    # 添加向量
    index.add_with_ids(embeddings, ids)

    # 保存索引
    faiss.write_index(index, os.path.join(output_dir, "embeddings_index.faiss"))

    # 加载索引（之后使用时）
    index = faiss.read_index(os.path.join(output_dir, "embeddings_index.faiss"))

    # 查询
    xrd_emb_dir = '/mnt/minio/battery/xrd/datasets/contrastive_learning/peaks_epoch44/xrd'
    cnt = 0
    hit_cnt =0
    for filename in os.listdir(xrd_emb_dir):
        if filename.endswith('.npy'):
            query_vector = np.load(os.path.join(xrd_emb_dir, filename))
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            k = 10  # 返回top10chang shichangshi
            distances, indices = index.search(query_vector, k)
            qid = int(filename.replace('mp-', '').replace('.npy', ''))
            if qid in indices:
                hit_cnt += 1
            cnt += 1
    print('cnt: ', cnt)
    print('hit cnt: ', hit_cnt)
    print('hit rate: ', round(hit_cnt / cnt, 5))
    