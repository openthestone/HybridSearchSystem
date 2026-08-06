cd /root/xihe_v1

SH=frnew_syn_v2/runs/shard2_den0

DATASET_FILE=/root/xihe_v1/frnew_syn_v2/dataset.bin \
DEVICE_IDS=0,1 \
SHARD_INDEX_DIRS=$SH/shard0/index,$SH/shard1/index \
WORK_DIR=/root/xihe_v1/frnew_syn_v2/runs/seg131072_den0/work \
NUM_QUERIES=0 \
RECALL_REF_FILE=/root/xihe_v1/$SH/recall_ref.bin RECALL_REF_THREADS=0 \
FULL_RECALL_TEXT_FILTER_BLOCK_DIM=16 \
NPU_TOPK_LOOP_COUNT=48 \
NOTIFY=1 \
./run.sh --compile 0 --convert-query 0 --profile 1 --repeat 3
