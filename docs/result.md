## block_dim
[Step] Running full-recall search sweep over FULL_RECALL_TEXT_FILTER_BLOCK_DIM_VALUES...
[Sweep] FULL_RECALL_TEXT_FILTER_BLOCK_DIM=16
[RESULT] latency ms: avg=35.2056 p50=36.7488 p90=48.0454 p99=59.7314 max=109.4479 (n=10888)
[RESULT] q0 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q1 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q2 recall=0.9983 (ref=3500, engine=3500)
[RESULT] q3 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q4 recall=0.9994 (ref=3500, engine=3500)
[RESULT] q5 recall=0.9997 (ref=3500, engine=3500)
[RESULT] q6 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q7 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q8 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q9 recall=1.0000 (ref=3500, engine=3500)
[RESULT] CPU-recall over 20 queries: avg=99.92%
[Done] Search log: /opt/huawei/dqj50056951/hx_npu/result/log/fr_search_block_dim_16.log


[Sweep] FULL_RECALL_TEXT_FILTER_BLOCK_DIM=24
[RESULT] latency ms: avg=33.6412 p50=35.0962 p90=46.0351 p99=57.5967 max=107.3235 (n=10888)
[RESULT] q0 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q1 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q2 recall=0.9983 (ref=3500, engine=3500)
[RESULT] q3 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q4 recall=0.9994 (ref=3500, engine=3500)
[RESULT] q5 recall=0.9997 (ref=3500, engine=3500)
[RESULT] q6 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q7 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q8 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q9 recall=1.0000 (ref=3500, engine=3500)
[RESULT] CPU-recall over 20 queries: avg=99.92%
[Done] Search log: /opt/huawei/dqj50056951/hx_npu/result/log/fr_search_block_dim_24.log


[Sweep] FULL_RECALL_TEXT_FILTER_BLOCK_DIM=32
[RESULT] latency ms: avg=32.7903 p50=34.2284 p90=44.9380 p99=56.4875 max=106.9812 (n=10888)
[RESULT] q0 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q1 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q2 recall=0.9983 (ref=3500, engine=3500)
[RESULT] q3 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q4 recall=0.9994 (ref=3500, engine=3500)
[RESULT] q5 recall=0.9997 (ref=3500, engine=3500)
[RESULT] q6 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q7 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q8 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q9 recall=1.0000 (ref=3500, engine=3500)
[RESULT] CPU-recall over 20 queries: avg=99.92%
[Done] Search log: /opt/huawei/dqj50056951/hx_npu/result/log/fr_search_block_dim_32.log


[Sweep] FULL_RECALL_TEXT_FILTER_BLOCK_DIM=40
[RESULT] latency ms: avg=32.1284 p50=33.4009 p90=44.0960 p99=56.2571 max=106.9566 (n=10888)
[RESULT] q0 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q1 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q2 recall=0.9983 (ref=3500, engine=3500)
[RESULT] q3 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q4 recall=0.9994 (ref=3500, engine=3500)
[RESULT] q5 recall=0.9997 (ref=3500, engine=3500)
[RESULT] q6 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q7 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q8 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q9 recall=1.0000 (ref=3500, engine=3500)
[RESULT] CPU-recall over 20 queries: avg=99.92%
[Done] Search log: /opt/huawei/dqj50056951/hx_npu/result/log/fr_search_block_dim_40.log
[Done] hx_npu run completed.

## density

固定 block_dim  = 40

### 0.01
[RESULT] latency ms: avg=29.9271 p50=30.9967 p90=41.8294 p99=53.4627 max=103.8222 (n=10888)
[RESULT] q0 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q1 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q2 recall=0.9983 (ref=3500, engine=3500)
[RESULT] q3 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q4 recall=0.9994 (ref=3500, engine=3500)
[RESULT] q5 recall=0.9997 (ref=3500, engine=3500)
[RESULT] q6 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q7 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q8 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q9 recall=1.0000 (ref=3500, engine=3500)
[RESULT] CPU-recall over 20 queries: avg=99.92%

### 0.02
[RESULT] latency ms: avg=30.2501 p50=31.3905 p90=42.1094 p99=53.5749 max=103.4822 (n=10888)
[RESULT] q0 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q1 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q2 recall=0.9983 (ref=3500, engine=3500)
[RESULT] q3 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q4 recall=0.9994 (ref=3500, engine=3500)
[RESULT] q5 recall=0.9997 (ref=3500, engine=3500)
[RESULT] q6 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q7 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q8 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q9 recall=1.0000 (ref=3500, engine=3500)

[RESULT] CPU-recall over 20 queries: avg=99.92%
### 0.05
[RESULT] latency ms: avg=32.4359 p50=33.8242 p90=44.5062 p99=56.0097 max=106.2974 (n=10888)
[RESULT] q0 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q1 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q2 recall=0.9983 (ref=3500, engine=3500)
[RESULT] q3 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q4 recall=0.9994 (ref=3500, engine=3500)
[RESULT] q5 recall=0.9997 (ref=3500, engine=3500)
[RESULT] q6 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q7 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q8 recall=1.0000 (ref=3500, engine=3500)
[RESULT] q9 recall=1.0000 (ref=3500, engine=3500)
[RESULT] CPU-recall over 20 queries: avg=99.92%
