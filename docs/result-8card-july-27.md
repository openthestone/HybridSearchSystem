[LOAD] shard 0 on device 0: docNum=1250000 segments=5 in 15735 ms
[LOAD] shard 1 on device 1: docNum=1250000 segments=5 in 14921 ms
[LOAD] shard 2 on device 2: docNum=1250000 segments=5 in 14871 ms
[LOAD] shard 3 on device 3: docNum=1250000 segments=5 in 14986 ms
[LOAD] shard 4 on device 4: docNum=1250000 segments=5 in 15036 ms
[LOAD] shard 5 on device 5: docNum=1250000 segments=5 in 14988 ms
[LOAD] shard 6 on device 6: docNum=1250000 segments=5 in 14980 ms
[LOAD] shard 7 on device 7: docNum=1250000 segments=5 in 15001 ms
[LOAD] 8 shard(s), 10000000 docs total, loaded in 120521 ms (cold start)

[RESULT] latency ms: avg=1.8644 p50=1.7644 p75=1.8507 p90=1.9327 p95=1.9883 p99=2.7417 max=94.6376 (n=10888)
[RESULT] CPU-recall over 10471 queries: avg=99.96%
[RESULT] 3874 per-query recall lines suppressed (--profile 2); see the CPU-recall summary

==== per-stage timing (avg per query, sorted by avg desc) ====
      avg_us       total_us      count  stage
      1488.4      129941227      87304  BatchSearch
       592.7       51745638      87304  Aggregator all
       382.7       33410286      87304  TextFilter BatchCompute
       358.5       31299083      87304  TextFilter_BitmapTextFilter_all
       263.4       22994710      87304  TextFilter_BitmapTextFilter_Kernel
       225.1        1376042       6112  AggrAndTopK_Aggregator_NPU_Memcpy
       181.1       15812050      87304  VectorScorerMmad Sync
       159.3       13907375      87304  VectorScorerMmad Kernel Sync
       139.4       12174314      87304  TextFilter BatchPrepareExpr
       135.7       10563940      77853  AggrAndTopK_TopK_NPU
       125.9       10992500      87304  TextFilter_Compute_GetPostOrderExpression
        68.0        5935499      87304  VectorScorerMmad Launch
        67.5        5665474      83965  AggrAndTopK_fill_d2h_copy
        66.7        5820564      87304  AggrAndTopK_Aggregator_NPU
        63.5        5329897      83965  AggrAndTopK_std_sort_sort
        61.7        5386330      87304  TextFilter BatchPreparePostings
        49.7        4337633      87304  TextFilter_PostingBitListToSet
        46.5        3906563      83965  AggrAndTopK_create_result
        21.3        1857523      87304  TextFilter_BitmapTextFilter_postExpr_malloc_h2d
        18.4        1606354      87304  VectorScorerMmad Kernel Launch
        15.0        1257965      83965  AggrAndTopK_mallocHost
        10.5         877853      83965  AggrAndTopK_FreeAggregator
         9.2         774630      83965  AggrAndTopK_fill_build
         3.2         276003      87304  TextFilter_BitmapTextFilter_teardown_free
         0.0           1766     261912  AggrAndTopK_allocBlocks
         0.0           1743      87304  TextFilter_BitmapTextFilter_resultStack_allocate

==== per-stage timing percentiles (us, over all invocations, sorted by p99) ====
       p50        p90        p99        max     count  stage
      1395       1562       1727      94353     87304  BatchSearch
       540        637        721      93439     87304  Aggregator all
       357        477        580      88857     87304  TextFilter BatchCompute
       335        453        554      88825     87304  TextFilter_BitmapTextFilter_all
       268        381        478        821     87304  TextFilter_BitmapTextFilter_Kernel
       172        282        335      88555     87304  VectorScorerMmad Sync
       158        270        319        723     87304  VectorScorerMmad Kernel Sync
       216        270        283       1176      6112  AggrAndTopK_Aggregator_NPU_Memcpy
       138        205        273      88030     87304  TextFilter BatchPrepareExpr
       128        193        258       1117     87304  TextFilter_Compute_GetPostOrderExpression
       130        180        232        562     77853  AggrAndTopK_TopK_NPU
        56        125        193        594     83965  AggrAndTopK_fill_d2h_copy
        65         87        118       1659     87304  VectorScorerMmad Launch
        57         80        109      87944     87304  TextFilter BatchPreparePostings
        49         70         94        858     87304  TextFilter_PostingBitListToSet
        65         79         88        232     83965  AggrAndTopK_std_sort_sort
        66         73         82        594     87304  AggrAndTopK_Aggregator_NPU
        41         69         80        205     83965  AggrAndTopK_create_result
        12         30         55        286     83965  AggrAndTopK_mallocHost
         9         15         28        192     83965  AggrAndTopK_FreeAggregator
        21         24         27        400     87304  TextFilter_BitmapTextFilter_postExpr_malloc_h2d
        18         21         27       1463     87304  VectorScorerMmad Kernel Launch
         9         12         15        348     83965  AggrAndTopK_fill_build
         3          4          4         46     87304  TextFilter_BitmapTextFilter_teardown_free
         0          0          0         75     87304  TextFilter_BitmapTextFilter_resultStack_allocate
         0          0          0         40    261912  AggrAndTopK_allocBlocks
