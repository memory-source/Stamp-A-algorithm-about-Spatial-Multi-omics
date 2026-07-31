#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
for i in 1 2 3 4 5; do
    echo "========== Running v7a Dataset $i =========="
    python /data/lvyongji/Assignment5/code/run_stamp_v7a.py --dataset $i > /data/lvyongji/Assignment5/code/v7a_d${i}.log 2>&1
    echo "Dataset $i finished"
done
echo "ALL DONE"
