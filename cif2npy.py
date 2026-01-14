import numpy as np
from chgnet.model import CHGNet
from pymatgen.core import Structure
from tqdm import tqdm

import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

CIF_DIR = '/mnt/minio/battery/xrd/datasets/raw_data/mp_cif'
OUTPUT_DIR = '/mnt/minio/battery/xrd/datasets/raw_data/mp_cif_npy'

def process_cif_file(file_path, model):
    if os.path.exists(os.path.join(OUTPUT_DIR, file_path.replace('.cif', '.npy'))):
        return
    try:
        cifpath = os.path.join(CIF_DIR, file_path)
        structure = Structure.from_file(cifpath)
        result = model.predict_structure(
            structure,
            task='e',  # 只需要能量任务来获取特征
            return_atom_feas=True,  # 关键参数：返回原子特征
            return_crystal_feas=False  # 不需要晶体级特征
        )

        np.save(os.path.join(OUTPUT_DIR, file_path.replace('.cif', '.npy')), result['atom_fea'])
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

def parallel_process(model, folder_path, max_workers=None):
    model = CHGNet.load()
    """使用线程池并行处理"""
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return
    
    # 获取所有文件路径
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(file)
    
    print(f"找到 {len(file_paths)} 个文件")
    results = []
    
    # 使用线程池并行处理
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_cif_file, file_path, model): file_path 
                  for file_path in file_paths}
        
        
        
        # 使用tqdm显示进度条
        with tqdm(total=len(file_paths), desc="处理文件中", unit="文件") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
                # 显示处理速度
                pbar.set_postfix({
                    "速度": f"{len(results)/(time.time()-start_time):.1f}文件/秒"
                })
        

    
    elapsed_time = time.time() - start_time
    print(f"线程池处理完成，耗时: {elapsed_time:.2f}秒")
    return results



if __name__ == "__main__":
    model = CHGNet.load()
    parallel_process(model, CIF_DIR, max_workers=10)


# embeddings = np.load(cifpath.replace(".cif", ".npy")) 
# print(embeddings.shape)