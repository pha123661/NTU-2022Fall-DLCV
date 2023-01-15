'''
1. put all submissions in ./all_submissions
'''

import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

random.seed(9453)

# ['1', '2', '3', '4', '5']
data_root, all_submissions, _ = next(iter(os.walk('./all_submissions')))
data_root = Path(data_root)
all_submissions = [data_root / sub for sub in sorted(all_submissions)]

results = defaultdict(list)


for submission in all_submissions:
    print(f"Reading '{submission}'...")
    for file in tqdm(submission.iterdir()):
        with file.open('r') as fp:
            preds = [int(x.strip()) for x in fp.readlines()]
            results[file.name].append(preds)

assert all(len(v) == len(next(iter(results.values()))) for v in results.values(
)), "read different length of files!"

output_path = Path('./ensemble_results')
output_path.mkdir(parents=True, exist_ok=True)
print(f"Writing files to '{output_path}'")
for filename, all_preds in tqdm(results.items()):
    pred_this_file = []
    for preds in zip(*all_preds):
        # choose maximum
        cnt = Counter(preds)
        voted = max(cnt, key=lambda key: cnt[key] + random.random())
        pred_this_file.append(voted)
    np.savetxt(output_path / filename, pred_this_file, fmt="%d")
print(f"Wrote {len(results)} files")
