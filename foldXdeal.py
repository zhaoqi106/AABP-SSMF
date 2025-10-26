import pandas as pd
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# === 配置路径 ===
csv_path = "../CSV/skempi2.csv"
foldx_path = "../../FOLDX/foldx.exe"
pdb_dir = "../PDBSOURCE/skempi2"
output_dir = "./skempi2"
output_failures = os.path.join(pdb_dir, "failed_mutations.txt")

# 创建所需目录
os.makedirs(output_dir, exist_ok=True)
temp_base_dir = os.path.join(pdb_dir, "_foldx_tmp")
os.makedirs(temp_base_dir, exist_ok=True)

failures = []
lock = Lock()

def process_sample(index, row):
    pdb_tag = row['pdb']
    mutation_str = row['Mutation']
    pdb_id = pdb_tag[:4]
    pdb_file = f"{pdb_id}.pdb"
    source_pdb_path = os.path.join(pdb_dir, pdb_file)
    temp_dir = os.path.join(temp_base_dir, f"tmp_{pdb_tag}")

    try:
        if not os.path.exists(source_pdb_path):
            raise FileNotFoundError("Missing PDB file")

        # 解析 mutation 字符串
        mutations = []
        for mut in mutation_str.split(','):
            chain, detail = mut.split(':')
            wt = detail[0]
            resnum = ''.join(filter(str.isdigit, detail[1:-1]))
            to_aa = detail[-1]
            mutations.append(f"{wt}{chain}{resnum}{to_aa};")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir)

        shutil.copy2(source_pdb_path, os.path.join(temp_dir, pdb_file))

        with open(os.path.join(temp_dir, "individual_list.txt"), 'w') as f:
            for m in mutations:
                f.write(m + "\n")

        cmd = f'"{foldx_path}" --command=BuildModel --pdb={pdb_file} --mutant-file=individual_list.txt --numberOfRuns=1'
        subprocess.run(cmd, shell=True, cwd=temp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        result_file = os.path.join(temp_dir, f"{pdb_id}_1.pdb")
        if os.path.exists(result_file):
            shutil.move(result_file, os.path.join(output_dir, f"{pdb_tag}.pdb"))
        else:
            raise RuntimeError("No output structure generated")

    except Exception as e:
        with lock:
            failures.append(f"{pdb_tag}: {e}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    df = pd.read_csv(csv_path)
    total = len(df)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_sample, idx, row): idx for idx, row in df.iterrows()}
        for _ in tqdm(as_completed(futures), total=total, desc="⏳ Running FoldX mutations", ncols=100):
            pass

    if failures:
        with open(output_failures, 'w') as f:
            for line in failures:
                f.write(line + '\n')
        print(f"\n[❌] 部分任务失败，已记录至: {output_failures}")
    else:
        print("\n[✅] 所有突变任务成功完成。")

if __name__ == "__main__":
    main()
