"""Run STAMP v6.1 on all 5 simulated datasets and report results."""
import os
import sys
import subprocess

results = {}

for i in range(1, 6):
    print(f"\n{'='*60}")
    print(f"Running STAMP on Simulated Dataset {i}")
    print(f"{'='*60}\n")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '1'
    
    cmd = f"cd /data/lvyongji/Assignment5/code && python run_stamp.py --dataset {i}"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Parse ARI from output
    ari = None
    nmi = None
    for line in result.stdout.split('\n'):
        if '[RESULT]' in line and 'ARI=' in line:
            parts = line.split('ARI=')[1].split(',')[0]
            ari = float(parts)
            parts_nmi = line.split('NMI=')[1].split(']')[0].split()[0]
            nmi = float(parts_nmi)
            break
    
    results[f'Dataset_{i}'] = {'ARI': ari, 'NMI': nmi}
    print(f"Dataset {i} -> ARI={ari}, NMI={nmi}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for k, v in results.items():
    print(f"{k}: ARI={v['ARI']:.4f}, NMI={v['NMI']:.4f}")
