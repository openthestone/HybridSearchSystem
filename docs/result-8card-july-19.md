## 串行

[Step] Index on disk:
  3.1G  /opt/huawei/dqj50056951/hx_npu/runs/8card_seg131072/work/index/shard0
  4.4G  /opt/huawei/dqj50056951/hx_npu/runs/8card_seg131072/work/index/shard1
  4.4G  /opt/huawei/dqj50056951/hx_npu/runs/8card_seg131072/work/index/shard2
  4.5G  /opt/huawei/dqj50056951/hx_npu/runs/8card_seg131072/work/index/shard3
  4.5G  /opt/huawei/dqj50056951/hx_npu/runs/8card_seg131072/work/index/shard4
  4.5G  /opt/huawei/dqj50056951/hx_npu/runs/8card_seg131072/work/index/shard5
  4.5G  /opt/huawei/dqj50056951/hx_npu/runs/8card_seg131072/work/index/shard6
  4.5G  /opt/huawei/dqj50056951/hx_npu/runs/8card_seg131072/work/index/shard7
[Step] Running full-recall search...
[Step] NPU memory before the search:
  NPU        HBM(MB)    used  processes                not-ours
  0      2866/32768     8.7%  -                        2866 MB
  1      2865/32768     8.7%  -                        2865 MB
  2      2851/32768     8.7%  -                        2851 MB
  3      2850/32768     8.7%  -                        2850 MB
  4      5442/32768    16.6%  gunicorn(2492MB)         2950 MB
  5      4918/32768    15.0%  gunicorn(1966MB)         2952 MB
  6      2864/32768     8.7%  -                        2864 MB
  7      2864/32768     8.7%  -                        2864 MB
  (note: npu-smi ids are physical; DEVICE_IDS are ACL logical ids and may differ)

[LOAD] shard 0 on device 0: docNum=1250000 segments=10 in 10103 ms
[LOAD] shard 1 on device 1: docNum=1250000 segments=10 in 13543 ms
[LOAD] shard 2 on device 2: docNum=1250000 segments=10 in 13760 ms
[LOAD] shard 3 on device 3: docNum=1250000 segments=10 in 13944 ms
[LOAD] shard 4 on device 4: docNum=1250000 segments=10 in 13956 ms
[LOAD] shard 5 on device 5: docNum=1250000 segments=10 in 13949 ms
[LOAD] shard 6 on device 6: docNum=1250000 segments=10 in 13987 ms
[LOAD] shard 7 on device 7: docNum=1250000 segments=10 in 14148 ms
[LOAD] 8 shard(s), 10000000 docs total, loaded in 107394 ms (cold start)
[RESULT] latency ms: avg=2.6571 p50=2.6915 p75=2.8258 p90=2.9483 p95=3.0230 p99=3.3388 max=73.1576 (n=10888)
[RESULT] CPU-recall over 10471 queries: avg=99.96%
[RESULT] 3862 per-query recall lines suppressed (--profile 2); see the CPU-recall summary

==== per-stage timing (avg per query, sorted by avg desc) ====
      avg_us       total_us      count  stage
      2013.0      175417761      87144  BatchSearch
       931.2       81146389      87144  Aggregator all
       516.6       45022476      87144  TextFilter BatchCompute
       330.2       28771684      87144  TextFilter_BitmapTextFilter_all
       276.9       23205560      83805  AggrAndTopK_std_sort_sort
       257.8        1976290       7666  AggrAndTopK_Aggregator_NPU_Memcpy
       214.4       18686753      87144  VectorScorerMmad Sync
       193.1       16825327      87144  TextFilter_BitmapTextFilter_Kernel
       189.5       16517239      87144  VectorScorerMmad Kernel Sync
       160.2       12200999      76139  AggrAndTopK_TopK_NPU
       135.5       11805298      87144  TextFilter BatchPrepareExpr
       118.4       10316926      87144  TextFilter_Compute_GetPostOrderExpression
       112.2        9400131      83805  AggrAndTopK_fill_d2h_emplace
        96.5        8410184      87144  VectorScorerMmad Launch
        73.8        6435459      87144  TextFilter_PostingBitListToSet
        67.1        5849789      87144  AggrAndTopK_Aggregator_NPU
        58.4        4897836      83805  AggrAndTopK_create_result
        56.5        4731294      83805  AggrAndTopK_fill_d2h_copy
        28.2        2366559      83805  AggrAndTopK_fill_build
        25.6        2227932      87144  TextFilter_BitmapTextFilter_postExpr_malloc_h2d
        19.9        1734304      87144  VectorScorerMmad Kernel Launch
        19.0        1594966      83805  AggrAndTopK_mallocHost
        16.2        1360542      83805  AggrAndTopK_FreeAggregator
         4.1         357420      87144  TextFilter_BitmapTextFilter_teardown_free
         0.3          27999      87144  TextFilter_BitmapTextFilter_resultStack_allocate
         0.2          52812     261432  AggrAndTopK_allocBlocks
         0.2          20653      87144  TextFilter_Compute_teardown_aclrtFree_devicePostings
         0.0           3716      87144  TextFilter_Compute_teardown_FreeBlock_bitset

==== per-stage timing percentiles (us, over all invocations, sorted by p99) ====
       p50        p90        p99        max     count  stage
      2023       2312       2508      72752     87144  BatchSearch
       949       1127       1297      71517     87144  Aggregator all
       500        688        851      71256     87144  TextFilter BatchCompute
       317        447        583      71113     87144  TextFilter_BitmapTextFilter_all
       218        292        354       2697     87144  VectorScorerMmad Sync
       196        275        354       4250     87144  TextFilter_BitmapTextFilter_Kernel
       302        324        341       2874     83805  AggrAndTopK_std_sort_sort
       194        265        322       2024     87144  VectorScorerMmad Kernel Sync
       247        304        320       2146      7666  AggrAndTopK_Aggregator_NPU_Memcpy
       131        200        307      13195     87144  TextFilter BatchPrepareExpr
       154        213        290       1916     76139  AggrAndTopK_TopK_NPU
       116        182        283      13178     87144  TextFilter_Compute_GetPostOrderExpression
       105        165        240       1376     83805  AggrAndTopK_fill_d2h_emplace
        87        143        192       2431     87144  VectorScorerMmad Launch
        40        100        180       1335     83805  AggrAndTopK_fill_d2h_copy
        50         95        173       9494     83805  AggrAndTopK_create_result
        74        103        139       1558     87144  TextFilter_PostingBitListToSet
        67         73         82       1290     87144  AggrAndTopK_Aggregator_NPU
        13         41         82       1836     83805  AggrAndTopK_mallocHost
        26         43         73        978     83805  AggrAndTopK_fill_build
        14         25         55        348     83805  AggrAndTopK_FreeAggregator
        26         29         33       2744     87144  TextFilter_BitmapTextFilter_postExpr_malloc_h2d
        18         25         32       2126     87144  VectorScorerMmad Kernel Launch
         4          5          6       1587     87144  TextFilter_BitmapTextFilter_teardown_free
         0          1          1        146     87144  TextFilter_BitmapTextFilter_resultStack_allocate
         0          1          1         55    261432  AggrAndTopK_allocBlocks
         0          1          1         33     87144  TextFilter_Compute_teardown_aclrtFree_devicePostings
         0          0          1         56     87144  TextFilter_Compute_teardown_FreeBlock_bitset

## 并行
[Step] Running full-recall search...
[Step] NPU memory before the search:
  NPU        HBM(MB)    used  processes                not-ours
  0      2870/32768     8.8%  -                        2870 MB
  1      2870/32768     8.8%  -                        2870 MB
  2      2856/32768     8.7%  -                        2856 MB
  3      2857/32768     8.7%  -                        2857 MB
  4      5441/32768    16.6%  gunicorn(2492MB)         2949 MB
  5      4919/32768    15.0%  gunicorn(1966MB)         2953 MB
  6      2866/32768     8.7%  -                        2866 MB
  7      2868/32768     8.8%  -                        2868 MB
  (note: npu-smi ids are physical; DEVICE_IDS are ACL logical ids and may differ)
[LOAD] shard 0 on device 0: docNum=1250000 segments=10 in 4676 ms
[LOAD] shard 1 on device 1: docNum=1250000 segments=10 in 6230 ms
[LOAD] shard 2 on device 2: docNum=1250000 segments=10 in 6214 ms
[LOAD] shard 3 on device 3: docNum=1250000 segments=10 in 10833 ms
[LOAD] shard 4 on device 4: docNum=1250000 segments=10 in 14037 ms
[LOAD] shard 5 on device 5: docNum=1250000 segments=10 in 14049 ms
[LOAD] shard 6 on device 6: docNum=1250000 segments=10 in 13909 ms
[LOAD] shard 7 on device 7: docNum=1250000 segments=10 in 13933 ms
[LOAD] 8 shard(s), 10000000 docs total, loaded in 83885 ms (cold start)
[RESULT] latency ms: avg=2.8571 p50=2.8165 p75=2.9949 p90=3.1450 p95=3.2423 p99=3.7006 max=189.4881 (n=10888)
[RESULT] CPU-recall over 10471 queries: avg=99.96%
[RESULT] 3858 per-query recall lines suppressed (--profile 2); see the CPU-recall summary

==== per-stage timing (avg per query, sorted by avg desc) ====
      avg_us       total_us      count  stage
      2198.5      191586512      87144  BatchSearch
       984.2       85769897      87144  TextFilter BatchCompute
       968.8       84421422      87144  Aggregator all
       588.8       51309756      87144  VectorScorerMmad BatchCompute
       487.9       42521570      87144  TextFilter_BitmapTextFilter_all
       332.4       28969655      87144  VectorScorerMmad Sync
       277.8       23282039      83805  AggrAndTopK_std_sort_sort
       265.7       23158207      87144  VectorScorerMmad Kernel Sync
       259.0        1985483       7666  AggrAndTopK_Aggregator_NPU_Memcpy
       227.6       19833533      87144  TextFilter_BitmapTextFilter_Kernel
       161.8       12321381      76139  AggrAndTopK_TopK_NPU
       158.7       13833083      87144  VectorScorerMmad Launch
       133.7       11648670      87144  TextFilter_Compute_GetPostOrderExpression
       115.0        9641331      83805  AggrAndTopK_fill_d2h_emplace
        78.1        6808200      87144  TextFilter_PostingBitListToSet
        69.7        6069741      87144  AggrAndTopK_Aggregator_NPU
        65.8        5518123      83805  AggrAndTopK_create_result
        54.7        4581079      83805  AggrAndTopK_fill_d2h_copy
        29.4        2460635      83805  AggrAndTopK_fill_build
        29.3        2549105      87144  TextFilter_BitmapTextFilter_postExpr_malloc_h2d
        22.2        1935922      87144  VectorScorerMmad Kernel Launch
        18.3        1533875      83805  AggrAndTopK_mallocHost
        16.1        1350448      83805  AggrAndTopK_FreeAggregator
         6.4         558596      87144  TextFilter_BitmapTextFilter_teardown_free
         0.4          34515      87144  TextFilter_BitmapTextFilter_resultStack_allocate
         0.3          28072      87144  TextFilter_Compute_teardown_aclrtFree_devicePostings
         0.2          60440     261432  AggrAndTopK_allocBlocks
         0.1           5584      87144  TextFilter_Compute_teardown_FreeBlock_bitset

==== per-stage timing percentiles (us, over all invocations, sorted by p99) ====
       p50        p90        p99        max     count  stage
      2160       2474       2733     189073     87144  BatchSearch
       932       1187       1420     187791     87144  TextFilter BatchCompute
       968       1160       1365     187943     87144  Aggregator all
       554        724        964      83999     87144  VectorScorerMmad BatchCompute
       453        645        878     187461     87144  TextFilter_BitmapTextFilter_all
       343        446        611      83913     87144  VectorScorerMmad Sync
       128        288        460       2875     87144  VectorScorerMmad Launch
       223        325        422       3107     87144  TextFilter_BitmapTextFilter_Kernel
       291        362        383       2535     87144  VectorScorerMmad Kernel Sync
       303        325        343       5669     83805  AggrAndTopK_std_sort_sort
       129        205        343      11411     87144  TextFilter_Compute_GetPostOrderExpression
       250        299        328       1906      7666  AggrAndTopK_Aggregator_NPU_Memcpy
       156        215        295       1730     76139  AggrAndTopK_TopK_NPU
       106        164        242      83792     83805  AggrAndTopK_fill_d2h_emplace
        54        106        197       8250     83805  AggrAndTopK_create_result
        41         95        171       1934     83805  AggrAndTopK_fill_d2h_copy
        78        108        151       2625     87144  TextFilter_PostingBitListToSet
        69         76         85       5598     87144  AggrAndTopK_Aggregator_NPU
        12         41         84        282     83805  AggrAndTopK_mallocHost
        26         46         77       1848     83805  AggrAndTopK_fill_build
        14         26         56       1176     83805  AggrAndTopK_FreeAggregator
        21         27         43       2541     87144  VectorScorerMmad Kernel Launch
        29         33         40        632     87144  TextFilter_BitmapTextFilter_postExpr_malloc_h2d
         6          8         12       2224     87144  TextFilter_BitmapTextFilter_teardown_free
         0          1          1       2197     87144  TextFilter_BitmapTextFilter_resultStack_allocate
         0          1          1        184     87144  TextFilter_Compute_teardown_aclrtFree_devicePostings
         0          1          1         51    261432  AggrAndTopK_allocBlocks
         0          0          1        119     87144  TextFilter_Compute_teardown_FreeBlock_bitset
[Done] Search log: /opt/huawei/dqj50056951/hx_npu/runs/8card_seg131072/result_parallel_p2/log/fr_search.log
[Step] NPU memory after the search:
  NPU        HBM(MB)    used  processes                not-ours
  0      2870/32768     8.8%  -                        2870 MB
  1      2867/32768     8.7%  -                        2867 MB
  2      2854/32768     8.7%  -                        2854 MB
  3      2857/32768     8.7%  -                        2857 MB
  4      5443/32768    16.6%  gunicorn(2492MB)         2951 MB
  5      4919/32768    15.0%  gunicorn(1966MB)         2953 MB
  6      2868/32768     8.8%  -                        2868 MB
  7      2868/32768     8.8%  -                        2868 MB
  (note: npu-smi ids are physical; DEVICE_IDS are ACL logical ids and may differ)
