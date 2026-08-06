## analyze（--top 400）
HUAWEI ORIGINAL-DATA DISTRIBUTION REPORT

### TAG_MAP  (35672 terms)
  tag_id range: 0 .. 35671   distinct ids: 35672
  sections (prefix before '#'), by #tags: 134 distinct
          5001  99072
          4146  99078
          4020  99036
          4020  99041
          2996  99009
          2971  99052
          2945  3000
          2133  2000
          1652  99097
          1064  99031
           770  3001
           370  99011
           369  99013
           368  2002
           366  10000
           342  99004
           337  1006
           222  99088
           205  10015
           187  99033
           184  99038
           184  99003
           143  99095
           142  99043
           117  99051
            80  10003
            56  99002
            24  99059
            24  99034
            20  99017
            15  99058
            14  99070
            10  99076
            10  99090
             8  99092
             7  317
             7  307
             6  305
             6  316
             6  300
             6  99069
             5  99057
             4  99074
             3  309
             3  1
             3  303
             2  99098
             2  99091
             2  99093
             2  99085
             2  310
             2  306
             2  8000
             2  99068
             2  308
             2  8006
             2  10014
             2  99065
             2  99055
             1  304
             1  99049
             1  99039
             1  99075
             1  10004
             1  1008
             1  99056
             1  99077
             1  99050
             1  1002
             1  10008
             1  1009
             1  99001
             1  301
             1  99042
             1  3002
             1  99066
             1  99064
             1  99014
             1  99062
             1  99054
             1  99045
             1  2004
             1  312
             1  8003
             1  99048
             1  99012
             1  99063
             1  99046
             1  10010
             1  2
             1  99010
             1  99005
             1  313
             1  99037
             1  315
             1  99053
             1  10012
             1  314
             1  99035
             1  99000
             1  1004
             1  99071
             1  8004
             1  10007
             1  99060
             1  1001
             1  10006
             1  99040
             1  110002
             1  10011
             1  99087
             1  2003
             1  110001
             1  1000
             1  99073
             1  10005
             1  10009
             1  10013
             1  99067
             1  1003
             1  99047
             1  311
             1  99061
             1  8010
             1  1007
             1  10002
             1  8005
             1  99032
             1  99018
             1  99028
             1  99044
             1  2001
             1  110000
             1  10001


### DATASET (HYDSET2)  dataset.bin
  version=2 doc_num=10000000 vector_dim=64 tag_num=35672
  stride(u64)=558  bytes/doc=4720  file=47.20GB expected=47.20GB match=OK
  sampling 20000 of 10000000 docs (seed=1234, numpy=yes)

-- per-doc tag count (how many tags each doc carries) --
  tags/doc: n=20000 min=142 p50=352 mean=378.4 p90=514 p99=1843 max=2525
    [       142,     340.6)      8223 #################################
    [     340.6,     539.2)      9927 ########################################
    [     539.2,     737.8)       373 ##
    [     737.8,     936.3)       408 ##
    [     936.3,      1135)       689 ###
    [      1135,      1334)       111 
    [      1334,      1532)        59 
    [      1532,      1731)         0 
    [      1731,      1929)       168 #
    [      1929,      2128)         0 
    [      2128,      2326)        35 
    [      2326,      2525)         7 
  density = avg/tag_num = 1.061%   (avg 378.4 of 35672 tags set per doc)

-- tag-frequency structure (fraction of sampled docs a tag appears in) --
  distinct tags seen in sample: 18682 of 35672  (never seen: 16990 => absent/very-rare)
  always-on (>=99.9%) : 79
  universal (90-99.9%): 28
  common    (5-90%)   : 894   <- dense bitset per segment; dominate filter cost
  low       (1-5%)    : 2080
  rare      (<1%)     : 15601   (+ 16990 never-seen)
  overall bitmap fill = set_bits/(docs*tags) = 1.061%
  set-bit concentration: top-20 tags carry 5.3% of all set bits, top-100=26.4%, top-1000=82.5%
  top-400 most-common tags (tag_id: fraction of docs, est. doc count):
    tag    471: 100.00%  (~10,000,000 docs)
    tag    470: 100.00%  (~10,000,000 docs)
    tag    469: 100.00%  (~10,000,000 docs)
    tag    468: 100.00%  (~10,000,000 docs)
    tag    467: 100.00%  (~10,000,000 docs)
    tag    466: 100.00%  (~10,000,000 docs)
    tag    465: 100.00%  (~10,000,000 docs)
    tag    464: 100.00%  (~10,000,000 docs)
    tag    463: 100.00%  (~10,000,000 docs)
    tag    462: 100.00%  (~10,000,000 docs)
    tag    461: 100.00%  (~10,000,000 docs)
    tag    460: 100.00%  (~10,000,000 docs)
    tag    459: 100.00%  (~10,000,000 docs)
    tag    457: 100.00%  (~10,000,000 docs)
    tag    456: 100.00%  (~10,000,000 docs)
    tag    455: 100.00%  (~10,000,000 docs)
    tag    454: 100.00%  (~10,000,000 docs)
    tag    453: 100.00%  (~10,000,000 docs)
    tag    452: 100.00%  (~10,000,000 docs)
    tag    451: 100.00%  (~10,000,000 docs)
    tag    449: 100.00%  (~10,000,000 docs)
    tag    448: 100.00%  (~10,000,000 docs)
    tag    446: 100.00%  (~10,000,000 docs)
    tag    445: 100.00%  (~10,000,000 docs)
    tag    442: 100.00%  (~10,000,000 docs)
    tag    441: 100.00%  (~10,000,000 docs)
    tag    440: 100.00%  (~10,000,000 docs)
    tag    439: 100.00%  (~10,000,000 docs)
    tag    438: 100.00%  (~10,000,000 docs)
    tag    437: 100.00%  (~10,000,000 docs)
    tag    436: 100.00%  (~10,000,000 docs)
    tag    434: 100.00%  (~10,000,000 docs)
    tag    433: 100.00%  (~10,000,000 docs)
    tag    432: 100.00%  (~10,000,000 docs)
    tag    431: 100.00%  (~10,000,000 docs)
    tag    430: 100.00%  (~10,000,000 docs)
    tag    424: 100.00%  (~10,000,000 docs)
    tag    423: 100.00%  (~10,000,000 docs)
    tag    422: 100.00%  (~10,000,000 docs)
    tag    420: 100.00%  (~10,000,000 docs)
    tag    419: 100.00%  (~10,000,000 docs)
    tag    418: 100.00%  (~10,000,000 docs)
    tag    411: 100.00%  (~10,000,000 docs)
    tag    408: 100.00%  (~10,000,000 docs)
    tag    403: 100.00%  (~10,000,000 docs)
    tag    392: 100.00%  (~10,000,000 docs)
    tag    391: 100.00%  (~10,000,000 docs)
    tag    388: 100.00%  (~10,000,000 docs)
    tag    387: 100.00%  (~10,000,000 docs)
    tag    386: 100.00%  (~10,000,000 docs)
    tag    385: 100.00%  (~10,000,000 docs)
    tag    384: 100.00%  (~10,000,000 docs)
    tag    380: 100.00%  (~10,000,000 docs)
    tag    376: 100.00%  (~10,000,000 docs)
    tag    374: 100.00%  (~10,000,000 docs)
    tag    372: 100.00%  (~10,000,000 docs)
    tag    222: 100.00%  (~10,000,000 docs)
    tag    220: 100.00%  (~10,000,000 docs)
    tag    218: 100.00%  (~10,000,000 docs)
    tag    217: 100.00%  (~10,000,000 docs)
    tag    195: 100.00%  (~10,000,000 docs)
    tag    194: 100.00%  (~10,000,000 docs)
    tag    193: 100.00%  (~10,000,000 docs)
    tag    192: 100.00%  (~10,000,000 docs)
    tag    181: 100.00%  (~10,000,000 docs)
    tag    180: 100.00%  (~10,000,000 docs)
    tag    179: 100.00%  (~10,000,000 docs)
    tag     27: 100.00%  (~10,000,000 docs)
    tag     25: 100.00%  (~10,000,000 docs)
    tag     14: 100.00%  (~10,000,000 docs)
    tag     13: 100.00%  (~10,000,000 docs)
    tag     12: 100.00%  (~10,000,000 docs)
    tag     10: 100.00%  (~10,000,000 docs)
    tag      9: 100.00%  (~10,000,000 docs)
    tag      8: 100.00%  (~10,000,000 docs)
    tag      7: 100.00%  (~10,000,000 docs)
    tag      6: 100.00%  (~10,000,000 docs)
    tag      0: 100.00%  (~10,000,000 docs)
    tag    413:  99.96%  (~9,996,000 docs)
    tag    409:  99.80%  (~9,980,500 docs)
    tag    458:  99.80%  (~9,980,000 docs)
    tag    450:  99.80%  (~9,980,000 docs)
    tag    447:  99.80%  (~9,980,000 docs)
    tag    444:  99.80%  (~9,980,000 docs)
    tag    443:  99.80%  (~9,980,000 docs)
    tag    435:  99.80%  (~9,980,000 docs)
    tag    429:  99.80%  (~9,980,000 docs)
    tag    428:  99.80%  (~9,980,000 docs)
    tag    427:  99.80%  (~9,980,000 docs)
    tag    426:  99.80%  (~9,980,000 docs)
    tag    425:  99.80%  (~9,980,000 docs)
    tag      1:  99.80%  (~9,980,000 docs)
    tag    406:  99.77%  (~9,977,000 docs)
    tag    375:  99.45%  (~9,945,000 docs)
    tag     21:  98.94%  (~9,893,500 docs)
    tag     20:  98.91%  (~9,891,500 docs)
    tag      2:  98.82%  (~9,882,000 docs)
    tag    373:  98.69%  (~9,868,500 docs)
    tag     26:  98.69%  (~9,868,500 docs)
    tag     22:  98.44%  (~9,844,000 docs)
    tag    416:  98.37%  (~9,836,500 docs)
    tag     19:  97.80%  (~9,779,500 docs)
    tag      3:  96.89%  (~9,689,000 docs)
    tag    405:  96.19%  (~9,618,500 docs)
    tag     23:  95.63%  (~9,563,000 docs)
    tag     18:  93.63%  (~9,363,000 docs)
    tag     28:  92.94%  (~9,293,500 docs)
    tag    410:  89.48%  (~8,948,000 docs)
    tag    545:  88.92%  (~8,892,500 docs)
    tag    544:  88.92%  (~8,892,500 docs)
    tag     17:  84.42%  (~8,442,500 docs)
    tag     16:  84.41%  (~8,441,500 docs)
    tag     15:  84.41%  (~8,441,500 docs)
    tag     24:  84.39%  (~8,439,500 docs)
    tag    396:  80.47%  (~8,046,500 docs)
    tag    390:  80.14%  (~8,013,500 docs)
    tag    196:  79.53%  (~7,952,500 docs)
    tag    478:  79.44%  (~7,944,000 docs)
    tag    407:  78.75%  (~7,875,000 docs)
    tag    507:  78.57%  (~7,857,499 docs)
    tag    506:  75.21%  (~7,521,000 docs)
    tag    378:  75.05%  (~7,504,999 docs)
    tag    397:  73.19%  (~7,319,000 docs)
    tag      5:  71.82%  (~7,181,999 docs)
    tag    393:  71.21%  (~7,121,499 docs)
    tag    401:  69.64%  (~6,964,000 docs)
    tag     11:  68.50%  (~6,850,000 docs)
    tag    480:  67.76%  (~6,776,000 docs)
    tag    395:  61.97%  (~6,197,000 docs)
    tag    398:  61.89%  (~6,189,000 docs)
    tag    533:  61.86%  (~6,186,000 docs)
    tag    508:  61.86%  (~6,186,000 docs)
    tag    550:  61.29%  (~6,128,500 docs)
    tag    377:  58.15%  (~5,815,000 docs)
    tag    421:  57.08%  (~5,708,000 docs)
    tag    389:  57.08%  (~5,708,000 docs)
    tag    542:  55.34%  (~5,534,500 docs)
    tag    546:  54.50%  (~5,449,500 docs)
    tag    415:  51.85%  (~5,185,000 docs)
    tag    206:  50.97%  (~5,096,500 docs)
    tag    186:  50.97%  (~5,096,500 docs)
    tag    213:  50.53%  (~5,053,000 docs)
    tag    190:  50.53%  (~5,053,000 docs)
    tag    208:  47.06%  (~4,706,500 docs)
    tag    184:  47.06%  (~4,706,500 docs)
    tag    202:  46.70%  (~4,670,000 docs)
    tag    191:  46.70%  (~4,670,000 docs)
    tag    400:  45.92%  (~4,592,000 docs)
    tag    399:  45.64%  (~4,564,000 docs)
    tag    394:  44.66%  (~4,465,500 docs)
    tag    551:  42.92%  (~4,292,000 docs)
    tag    504:  42.92%  (~4,292,000 docs)
    tag    219:  42.74%  (~4,274,000 docs)
    tag    417:  38.71%  (~3,871,500 docs)
    tag    894:  38.03%  (~3,803,000 docs)
    tag    221:  35.58%  (~3,558,000 docs)
    tag      4:  35.58%  (~3,558,000 docs)
    tag    271:  35.08%  (~3,508,000 docs)
    tag    150:  35.08%  (~3,508,000 docs)
    tag    368:  35.08%  (~3,507,500 docs)
    tag    152:  35.08%  (~3,507,500 docs)
    tag    234:  35.04%  (~3,504,000 docs)
    tag    151:  35.04%  (~3,504,000 docs)
    tag    298:  35.03%  (~3,503,500 docs)
    tag     63:  35.03%  (~3,503,500 docs)
    tag    249:  35.02%  (~3,502,500 docs)
    tag    166:  35.02%  (~3,502,500 docs)
    tag    232:  35.02%  (~3,502,000 docs)
    tag    102:  35.02%  (~3,502,000 docs)
    tag    236:  35.01%  (~3,501,000 docs)
    tag    143:  35.01%  (~3,501,000 docs)
    tag    344:  35.00%  (~3,500,000 docs)
    tag    154:  35.00%  (~3,500,000 docs)
    tag    246:  34.98%  (~3,498,500 docs)
    tag    113:  34.98%  (~3,498,500 docs)
    tag    240:  34.95%  (~3,494,500 docs)
    tag     88:  34.95%  (~3,494,500 docs)
    tag    295:  34.92%  (~3,492,500 docs)
    tag    268:  34.92%  (~3,492,500 docs)
    tag    148:  34.92%  (~3,492,500 docs)
    tag     31:  34.92%  (~3,492,500 docs)
    tag    347:  34.92%  (~3,492,000 docs)
    tag    310:  34.92%  (~3,492,000 docs)
    tag    305:  34.92%  (~3,492,000 docs)
    tag    160:  34.92%  (~3,492,000 docs)
    tag    157:  34.92%  (~3,492,000 docs)
    tag     91:  34.92%  (~3,492,000 docs)
    tag    308:  34.91%  (~3,491,500 docs)
    tag    168:  34.91%  (~3,491,500 docs)
    tag    302:  34.91%  (~3,491,000 docs)
    tag    252:  34.91%  (~3,491,000 docs)
    tag    156:  34.91%  (~3,491,000 docs)
    tag    153:  34.91%  (~3,491,000 docs)
    tag    359:  34.91%  (~3,490,500 docs)
    tag    163:  34.91%  (~3,490,500 docs)
    tag    342:  34.89%  (~3,489,000 docs)
    tag     70:  34.89%  (~3,489,000 docs)
    tag    328:  34.88%  (~3,488,000 docs)
    tag     68:  34.88%  (~3,488,000 docs)
    tag    229:  34.83%  (~3,483,000 docs)
    tag     30:  34.83%  (~3,483,000 docs)
    tag    278:  34.83%  (~3,482,500 docs)
    tag    149:  34.83%  (~3,482,500 docs)
    tag    304:  34.79%  (~3,479,000 docs)
    tag     72:  34.79%  (~3,479,000 docs)
    tag    231:  34.75%  (~3,474,999 docs)
    tag    110:  34.75%  (~3,474,999 docs)
    tag    336:  34.72%  (~3,472,000 docs)
    tag    323:  34.72%  (~3,472,000 docs)
    tag    111:  34.72%  (~3,472,000 docs)
    tag     61:  34.72%  (~3,472,000 docs)
    tag    225:  34.71%  (~3,471,000 docs)
    tag    155:  34.71%  (~3,471,000 docs)
    tag    312:  34.71%  (~3,470,500 docs)
    tag     87:  34.71%  (~3,470,500 docs)
    tag    324:  34.66%  (~3,466,000 docs)
    tag    319:  34.66%  (~3,466,000 docs)
    tag     97:  34.66%  (~3,466,000 docs)
    tag     76:  34.66%  (~3,466,000 docs)
    tag    224:  34.62%  (~3,462,500 docs)
    tag    104:  34.62%  (~3,462,500 docs)
    tag    346:  34.62%  (~3,461,500 docs)
    tag     83:  34.62%  (~3,461,500 docs)
    tag    311:  34.61%  (~3,461,000 docs)
    tag     78:  34.61%  (~3,461,000 docs)
    tag    241:  34.61%  (~3,460,500 docs)
    tag     93:  34.61%  (~3,460,500 docs)
    tag    370:  34.30%  (~3,430,000 docs)
    tag     32:  34.30%  (~3,430,000 docs)
    tag   1205:  34.27%  (~3,426,500 docs)
    tag    269:  33.87%  (~3,386,500 docs)
    tag    107:  33.87%  (~3,386,500 docs)
    tag    584:  33.79%  (~3,379,000 docs)
    tag    369:  33.73%  (~3,372,500 docs)
    tag     45:  33.73%  (~3,372,500 docs)
    tag   1206:  31.61%  (~3,161,500 docs)
    tag    275:  31.46%  (~3,146,000 docs)
    tag     95:  31.46%  (~3,146,000 docs)
    tag    276:  31.25%  (~3,125,500 docs)
    tag    116:  31.25%  (~3,125,500 docs)
    tag    292:  31.23%  (~3,122,500 docs)
    tag    130:  31.23%  (~3,122,500 docs)
    tag    238:  31.19%  (~3,118,500 docs)
    tag     80:  31.19%  (~3,118,500 docs)
    tag    828:  30.74%  (~3,074,000 docs)
    tag    827:  30.72%  (~3,071,999 docs)
    tag    690:  30.70%  (~3,070,500 docs)
    tag    879:  30.69%  (~3,069,000 docs)
    tag    878:  30.68%  (~3,068,000 docs)
    tag    266:  30.68%  (~3,068,000 docs)
    tag    146:  30.68%  (~3,068,000 docs)
    tag    687:  30.68%  (~3,067,500 docs)
    tag    635:  30.67%  (~3,066,999 docs)
    tag    634:  30.67%  (~3,066,999 docs)
    tag    860:  30.66%  (~3,066,500 docs)
    tag    725:  30.66%  (~3,066,000 docs)
    tag    859:  30.65%  (~3,065,500 docs)
    tag    345:  30.65%  (~3,065,000 docs)
    tag    171:  30.65%  (~3,065,000 docs)
    tag    283:  30.64%  (~3,064,500 docs)
    tag    176:  30.64%  (~3,064,500 docs)
    tag    639:  30.64%  (~3,063,500 docs)
    tag    833:  30.63%  (~3,063,000 docs)
    tag    723:  30.63%  (~3,062,500 docs)
    tag    284:  30.63%  (~3,062,500 docs)
    tag    251:  30.63%  (~3,062,500 docs)
    tag    169:  30.63%  (~3,062,500 docs)
    tag    101:  30.63%  (~3,062,500 docs)
    tag    835:  30.62%  (~3,062,000 docs)
    tag    250:  30.62%  (~3,062,000 docs)
    tag    244:  30.62%  (~3,062,000 docs)
    tag    174:  30.62%  (~3,062,000 docs)
    tag    114:  30.62%  (~3,062,000 docs)
    tag    659:  30.61%  (~3,060,500 docs)
    tag    641:  30.60%  (~3,060,000 docs)
    tag    871:  30.58%  (~3,057,500 docs)
    tag    662:  30.57%  (~3,057,000 docs)
    tag    870:  30.56%  (~3,056,500 docs)
    tag    722:  30.55%  (~3,055,000 docs)
    tag    721:  30.55%  (~3,055,000 docs)
    tag    703:  30.55%  (~3,054,500 docs)
    tag    700:  30.54%  (~3,054,000 docs)
    tag    260:  30.54%  (~3,054,000 docs)
    tag    117:  30.54%  (~3,054,000 docs)
    tag    825:  30.53%  (~3,052,500 docs)
    tag    685:  30.53%  (~3,052,500 docs)
    tag    684:  30.53%  (~3,052,500 docs)
    tag    826:  30.51%  (~3,051,500 docs)
    tag    667:  30.50%  (~3,049,500 docs)
    tag    665:  30.50%  (~3,049,500 docs)
    tag    881:  30.48%  (~3,048,000 docs)
    tag    844:  30.48%  (~3,048,000 docs)
    tag    880:  30.47%  (~3,047,000 docs)
    tag    845:  30.47%  (~3,047,000 docs)
    tag    862:  30.46%  (~3,046,000 docs)
    tag    698:  30.46%  (~3,046,000 docs)
    tag    696:  30.46%  (~3,046,000 docs)
    tag    821:  30.45%  (~3,045,500 docs)
    tag    861:  30.45%  (~3,045,000 docs)
    tag    822:  30.45%  (~3,044,500 docs)
    tag    675:  30.44%  (~3,044,000 docs)
    tag    852:  30.42%  (~3,042,000 docs)
    tag    743:  30.42%  (~3,042,000 docs)
    tag    851:  30.41%  (~3,041,000 docs)
    tag    679:  30.40%  (~3,040,500 docs)
    tag    636:  30.39%  (~3,039,500 docs)
    tag    742:  30.39%  (~3,038,500 docs)
    tag    637:  30.39%  (~3,038,500 docs)
    tag    873:  30.36%  (~3,036,499 docs)
    tag    872:  30.35%  (~3,035,500 docs)
    tag    840:  30.35%  (~3,035,500 docs)
    tag    842:  30.34%  (~3,034,500 docs)
    tag    677:  30.33%  (~3,033,000 docs)
    tag    673:  30.33%  (~3,033,000 docs)
    tag    819:  30.31%  (~3,031,499 docs)
    tag    741:  30.31%  (~3,031,000 docs)
    tag    740:  30.31%  (~3,031,000 docs)
    tag    820:  30.30%  (~3,030,500 docs)
    tag    720:  30.28%  (~3,028,000 docs)
    tag    719:  30.27%  (~3,027,000 docs)
    tag    683:  30.25%  (~3,025,000 docs)
    tag    682:  30.24%  (~3,024,000 docs)
    tag    669:  30.23%  (~3,023,000 docs)
    tag    671:  30.22%  (~3,022,000 docs)
    tag    694:  30.19%  (~3,019,000 docs)
    tag    766:  30.19%  (~3,018,500 docs)
    tag    765:  30.19%  (~3,018,500 docs)
    tag    692:  30.18%  (~3,018,000 docs)
    tag    242:  30.15%  (~3,015,500 docs)
    tag     85:  30.15%  (~3,015,500 docs)
    tag    764:  30.11%  (~3,011,000 docs)
    tag    763:  30.11%  (~3,011,000 docs)
    tag    762:  30.09%  (~3,008,500 docs)
    tag    286:  30.08%  (~3,008,000 docs)
    tag     47:  30.08%  (~3,008,000 docs)
    tag    761:  30.08%  (~3,007,500 docs)
    tag    718:  30.05%  (~3,005,500 docs)
    tag    717:  30.05%  (~3,005,500 docs)
    tag    681:  30.04%  (~3,004,500 docs)
    tag    306:  30.04%  (~3,004,500 docs)
    tag     42:  30.04%  (~3,004,500 docs)
    tag    739:  30.04%  (~3,004,000 docs)
    tag    755:  30.04%  (~3,003,500 docs)
    tag    655:  30.04%  (~3,003,500 docs)
    tag    654:  30.04%  (~3,003,500 docs)
    tag    738:  30.03%  (~3,003,000 docs)
    tag    678:  30.01%  (~3,001,499 docs)
    tag    674:  30.01%  (~3,001,499 docs)
    tag    716:  30.00%  (~3,000,500 docs)
    tag    715:  30.00%  (~3,000,500 docs)
    tag    714:  29.99%  (~2,999,000 docs)
    tag    713:  29.98%  (~2,998,000 docs)
    tag    689:  29.98%  (~2,998,000 docs)
    tag    686:  29.98%  (~2,998,000 docs)
    tag    657:  29.97%  (~2,996,500 docs)
    tag    656:  29.97%  (~2,996,500 docs)
    tag    228:  29.97%  (~2,996,500 docs)
    tag    138:  29.97%  (~2,996,500 docs)
    tag    707:  29.95%  (~2,995,000 docs)
    tag    653:  29.95%  (~2,995,000 docs)
    tag    658:  29.94%  (~2,994,000 docs)
    tag    706:  29.94%  (~2,993,500 docs)
    tag    661:  29.93%  (~2,993,000 docs)
    tag    366:  29.93%  (~2,993,000 docs)
    tag    362:  29.93%  (~2,993,000 docs)
    tag    334:  29.93%  (~2,993,000 docs)
    tag    325:  29.93%  (~2,993,000 docs)
    tag    277:  29.93%  (~2,993,000 docs)
    tag    140:  29.93%  (~2,993,000 docs)
    tag    136:  29.93%  (~2,993,000 docs)
    tag    134:  29.93%  (~2,993,000 docs)
    tag    131:  29.93%  (~2,993,000 docs)
    tag     51:  29.93%  (~2,993,000 docs)
    tag    705:  29.93%  (~2,992,500 docs)
    tag    643:  29.91%  (~2,991,000 docs)
    tag    642:  29.91%  (~2,991,000 docs)
    tag    327:  29.91%  (~2,991,000 docs)
    tag     39:  29.91%  (~2,991,000 docs)
    tag    644:  29.89%  (~2,988,500 docs)
    tag    645:  29.88%  (~2,987,500 docs)
    tag    756:  29.87%  (~2,987,000 docs)
    tag    754:  29.87%  (~2,987,000 docs)
    tag    753:  29.87%  (~2,987,000 docs)
    tag    726:  29.87%  (~2,987,000 docs)
    tag    724:  29.86%  (~2,985,999 docs)
    tag    809:  29.84%  (~2,984,000 docs)
    tag    760:  29.84%  (~2,983,500 docs)
    tag    759:  29.84%  (~2,983,500 docs)
    tag    758:  29.83%  (~2,982,500 docs)
    tag    757:  29.83%  (~2,982,500 docs)
    tag    737:  29.81%  (~2,980,999 docs)
    tag    736:  29.81%  (~2,980,999 docs)
    tag    752:  29.79%  (~2,979,500 docs)
    tag    751:  29.78%  (~2,977,500 docs)
    tag    708:  29.77%  (~2,977,000 docs)
    tag    638:  29.77%  (~2,977,000 docs)
    tag    711:  29.76%  (~2,975,999 docs)
    tag    712:  29.75%  (~2,975,500 docs)
    tag    710:  29.75%  (~2,975,000 docs)
    tag    709:  29.75%  (~2,975,000 docs)
  rarest-400 SEEN tags (lowest non-zero frequency in the sample):
    tag    402:  0.0050%  (1/20000 sampled docs)
    tag    808:  0.0050%  (1/20000 sampled docs)
    tag    886:  0.0050%  (1/20000 sampled docs)
    tag    893:  0.0050%  (1/20000 sampled docs)
    tag    908:  0.0050%  (1/20000 sampled docs)
    tag    929:  0.0050%  (1/20000 sampled docs)
    tag    938:  0.0050%  (1/20000 sampled docs)
    tag    939:  0.0050%  (1/20000 sampled docs)
    tag    947:  0.0050%  (1/20000 sampled docs)
    tag    948:  0.0050%  (1/20000 sampled docs)
    tag    956:  0.0050%  (1/20000 sampled docs)
    tag    970:  0.0050%  (1/20000 sampled docs)
    tag    985:  0.0050%  (1/20000 sampled docs)
    tag   1050:  0.0050%  (1/20000 sampled docs)
    tag   1052:  0.0050%  (1/20000 sampled docs)
    tag   1057:  0.0050%  (1/20000 sampled docs)
    tag   1064:  0.0050%  (1/20000 sampled docs)
    tag   1075:  0.0050%  (1/20000 sampled docs)
    tag   1076:  0.0050%  (1/20000 sampled docs)
    tag   1132:  0.0050%  (1/20000 sampled docs)
    tag   1139:  0.0050%  (1/20000 sampled docs)
    tag   1152:  0.0050%  (1/20000 sampled docs)
    tag   1160:  0.0050%  (1/20000 sampled docs)
    tag   1161:  0.0050%  (1/20000 sampled docs)
    tag   1163:  0.0050%  (1/20000 sampled docs)
    tag   1169:  0.0050%  (1/20000 sampled docs)
    tag   1170:  0.0050%  (1/20000 sampled docs)
    tag   1171:  0.0050%  (1/20000 sampled docs)
    tag   1182:  0.0050%  (1/20000 sampled docs)
    tag   1183:  0.0050%  (1/20000 sampled docs)
    tag   1191:  0.0050%  (1/20000 sampled docs)
    tag   1199:  0.0050%  (1/20000 sampled docs)
    tag   1202:  0.0050%  (1/20000 sampled docs)
    tag   1211:  0.0050%  (1/20000 sampled docs)
    tag   1215:  0.0050%  (1/20000 sampled docs)
    tag   1216:  0.0050%  (1/20000 sampled docs)
    tag   1222:  0.0050%  (1/20000 sampled docs)
    tag   1229:  0.0050%  (1/20000 sampled docs)
    tag   1251:  0.0050%  (1/20000 sampled docs)
    tag   1260:  0.0050%  (1/20000 sampled docs)
    tag   1267:  0.0050%  (1/20000 sampled docs)
    tag   1269:  0.0050%  (1/20000 sampled docs)
    tag   1274:  0.0050%  (1/20000 sampled docs)
    tag   1275:  0.0050%  (1/20000 sampled docs)
    tag   1279:  0.0050%  (1/20000 sampled docs)
    tag   1280:  0.0050%  (1/20000 sampled docs)
    tag   1281:  0.0050%  (1/20000 sampled docs)
    tag   1283:  0.0050%  (1/20000 sampled docs)
    tag   1286:  0.0050%  (1/20000 sampled docs)
    tag   2004:  0.0050%  (1/20000 sampled docs)
    tag   2005:  0.0050%  (1/20000 sampled docs)
    tag   2008:  0.0050%  (1/20000 sampled docs)
    tag   2010:  0.0050%  (1/20000 sampled docs)
    tag   2019:  0.0050%  (1/20000 sampled docs)
    tag   2028:  0.0050%  (1/20000 sampled docs)
    tag   2061:  0.0050%  (1/20000 sampled docs)
    tag   2062:  0.0050%  (1/20000 sampled docs)
    tag   2068:  0.0050%  (1/20000 sampled docs)
    tag   2072:  0.0050%  (1/20000 sampled docs)
    tag   2074:  0.0050%  (1/20000 sampled docs)
    tag   2092:  0.0050%  (1/20000 sampled docs)
    tag   2098:  0.0050%  (1/20000 sampled docs)
    tag   2106:  0.0050%  (1/20000 sampled docs)
    tag   2110:  0.0050%  (1/20000 sampled docs)
    tag   2111:  0.0050%  (1/20000 sampled docs)
    tag   2112:  0.0050%  (1/20000 sampled docs)
    tag   2115:  0.0050%  (1/20000 sampled docs)
    tag   2117:  0.0050%  (1/20000 sampled docs)
    tag   2121:  0.0050%  (1/20000 sampled docs)
    tag   2122:  0.0050%  (1/20000 sampled docs)
    tag   2129:  0.0050%  (1/20000 sampled docs)
    tag   2137:  0.0050%  (1/20000 sampled docs)
    tag   2138:  0.0050%  (1/20000 sampled docs)
    tag   2140:  0.0050%  (1/20000 sampled docs)
    tag   2147:  0.0050%  (1/20000 sampled docs)
    tag   2150:  0.0050%  (1/20000 sampled docs)
    tag   2152:  0.0050%  (1/20000 sampled docs)
    tag   2155:  0.0050%  (1/20000 sampled docs)
    tag   2275:  0.0050%  (1/20000 sampled docs)
    tag   2279:  0.0050%  (1/20000 sampled docs)
    tag   2288:  0.0050%  (1/20000 sampled docs)
    tag   2290:  0.0050%  (1/20000 sampled docs)
    tag   2293:  0.0050%  (1/20000 sampled docs)
    tag   2294:  0.0050%  (1/20000 sampled docs)
    tag   2306:  0.0050%  (1/20000 sampled docs)
    tag   2307:  0.0050%  (1/20000 sampled docs)
    tag   2310:  0.0050%  (1/20000 sampled docs)
    tag   2311:  0.0050%  (1/20000 sampled docs)
    tag   2326:  0.0050%  (1/20000 sampled docs)
    tag   2339:  0.0050%  (1/20000 sampled docs)
    tag   2340:  0.0050%  (1/20000 sampled docs)
    tag   2353:  0.0050%  (1/20000 sampled docs)
    tag   2354:  0.0050%  (1/20000 sampled docs)
    tag   2356:  0.0050%  (1/20000 sampled docs)
    tag   2357:  0.0050%  (1/20000 sampled docs)
    tag   2358:  0.0050%  (1/20000 sampled docs)
    tag   2364:  0.0050%  (1/20000 sampled docs)
    tag   2365:  0.0050%  (1/20000 sampled docs)
    tag   2370:  0.0050%  (1/20000 sampled docs)
    tag   2372:  0.0050%  (1/20000 sampled docs)
    tag   2373:  0.0050%  (1/20000 sampled docs)
    tag   2376:  0.0050%  (1/20000 sampled docs)
    tag   2387:  0.0050%  (1/20000 sampled docs)
    tag   2406:  0.0050%  (1/20000 sampled docs)
    tag   2414:  0.0050%  (1/20000 sampled docs)
    tag   2420:  0.0050%  (1/20000 sampled docs)
    tag   2422:  0.0050%  (1/20000 sampled docs)
    tag   2426:  0.0050%  (1/20000 sampled docs)
    tag   2428:  0.0050%  (1/20000 sampled docs)
    tag   2429:  0.0050%  (1/20000 sampled docs)
    tag   2434:  0.0050%  (1/20000 sampled docs)
    tag   2439:  0.0050%  (1/20000 sampled docs)
    tag   2441:  0.0050%  (1/20000 sampled docs)
    tag   2443:  0.0050%  (1/20000 sampled docs)
    tag   2482:  0.0050%  (1/20000 sampled docs)
    tag   2483:  0.0050%  (1/20000 sampled docs)
    tag   2484:  0.0050%  (1/20000 sampled docs)
    tag   2485:  0.0050%  (1/20000 sampled docs)
    tag   2487:  0.0050%  (1/20000 sampled docs)
    tag   2489:  0.0050%  (1/20000 sampled docs)
    tag   2498:  0.0050%  (1/20000 sampled docs)
    tag   2501:  0.0050%  (1/20000 sampled docs)
    tag   2502:  0.0050%  (1/20000 sampled docs)
    tag   2509:  0.0050%  (1/20000 sampled docs)
    tag   2510:  0.0050%  (1/20000 sampled docs)
    tag   2525:  0.0050%  (1/20000 sampled docs)
    tag   2527:  0.0050%  (1/20000 sampled docs)
    tag   2534:  0.0050%  (1/20000 sampled docs)
    tag   2540:  0.0050%  (1/20000 sampled docs)
    tag   2556:  0.0050%  (1/20000 sampled docs)
    tag   2561:  0.0050%  (1/20000 sampled docs)
    tag   2563:  0.0050%  (1/20000 sampled docs)
    tag   2569:  0.0050%  (1/20000 sampled docs)
    tag   2666:  0.0050%  (1/20000 sampled docs)
    tag   2677:  0.0050%  (1/20000 sampled docs)
    tag   2678:  0.0050%  (1/20000 sampled docs)
    tag   2689:  0.0050%  (1/20000 sampled docs)
    tag   2690:  0.0050%  (1/20000 sampled docs)
    tag   2691:  0.0050%  (1/20000 sampled docs)
    tag   2705:  0.0050%  (1/20000 sampled docs)
    tag   2706:  0.0050%  (1/20000 sampled docs)
    tag   2707:  0.0050%  (1/20000 sampled docs)
    tag   2711:  0.0050%  (1/20000 sampled docs)
    tag   2718:  0.0050%  (1/20000 sampled docs)
    tag   2723:  0.0050%  (1/20000 sampled docs)
    tag   2724:  0.0050%  (1/20000 sampled docs)
    tag   2728:  0.0050%  (1/20000 sampled docs)
    tag   2731:  0.0050%  (1/20000 sampled docs)
    tag   2733:  0.0050%  (1/20000 sampled docs)
    tag   2735:  0.0050%  (1/20000 sampled docs)
    tag   2738:  0.0050%  (1/20000 sampled docs)
    tag   2742:  0.0050%  (1/20000 sampled docs)
    tag   2747:  0.0050%  (1/20000 sampled docs)
    tag   2753:  0.0050%  (1/20000 sampled docs)
    tag   2754:  0.0050%  (1/20000 sampled docs)
    tag   2761:  0.0050%  (1/20000 sampled docs)
    tag   2763:  0.0050%  (1/20000 sampled docs)
    tag   2765:  0.0050%  (1/20000 sampled docs)
    tag   2766:  0.0050%  (1/20000 sampled docs)
    tag   2767:  0.0050%  (1/20000 sampled docs)
    tag   2775:  0.0050%  (1/20000 sampled docs)
    tag   2782:  0.0050%  (1/20000 sampled docs)
    tag   2787:  0.0050%  (1/20000 sampled docs)
    tag   2790:  0.0050%  (1/20000 sampled docs)
    tag   2791:  0.0050%  (1/20000 sampled docs)
    tag   2794:  0.0050%  (1/20000 sampled docs)
    tag   2795:  0.0050%  (1/20000 sampled docs)
    tag   2801:  0.0050%  (1/20000 sampled docs)
    tag   2802:  0.0050%  (1/20000 sampled docs)
    tag   2803:  0.0050%  (1/20000 sampled docs)
    tag   2809:  0.0050%  (1/20000 sampled docs)
    tag   2813:  0.0050%  (1/20000 sampled docs)
    tag   2814:  0.0050%  (1/20000 sampled docs)
    tag   2815:  0.0050%  (1/20000 sampled docs)
    tag   2822:  0.0050%  (1/20000 sampled docs)
    tag   2826:  0.0050%  (1/20000 sampled docs)
    tag   2831:  0.0050%  (1/20000 sampled docs)
    tag   2836:  0.0050%  (1/20000 sampled docs)
    tag   2840:  0.0050%  (1/20000 sampled docs)
    tag   2842:  0.0050%  (1/20000 sampled docs)
    tag   2844:  0.0050%  (1/20000 sampled docs)
    tag   2847:  0.0050%  (1/20000 sampled docs)
    tag   2850:  0.0050%  (1/20000 sampled docs)
    tag   2852:  0.0050%  (1/20000 sampled docs)
    tag   2853:  0.0050%  (1/20000 sampled docs)
    tag   2854:  0.0050%  (1/20000 sampled docs)
    tag   2855:  0.0050%  (1/20000 sampled docs)
    tag   2856:  0.0050%  (1/20000 sampled docs)
    tag   2860:  0.0050%  (1/20000 sampled docs)
    tag   2861:  0.0050%  (1/20000 sampled docs)
    tag   2863:  0.0050%  (1/20000 sampled docs)
    tag   2866:  0.0050%  (1/20000 sampled docs)
    tag   2868:  0.0050%  (1/20000 sampled docs)
    tag   2876:  0.0050%  (1/20000 sampled docs)
    tag   2880:  0.0050%  (1/20000 sampled docs)
    tag   2885:  0.0050%  (1/20000 sampled docs)
    tag   2890:  0.0050%  (1/20000 sampled docs)
    tag   2894:  0.0050%  (1/20000 sampled docs)
    tag   2896:  0.0050%  (1/20000 sampled docs)
    tag   2903:  0.0050%  (1/20000 sampled docs)
    tag   2913:  0.0050%  (1/20000 sampled docs)
    tag   2915:  0.0050%  (1/20000 sampled docs)
    tag   2919:  0.0050%  (1/20000 sampled docs)
    tag   2922:  0.0050%  (1/20000 sampled docs)
    tag   2925:  0.0050%  (1/20000 sampled docs)
    tag   2929:  0.0050%  (1/20000 sampled docs)
    tag   2931:  0.0050%  (1/20000 sampled docs)
    tag   2946:  0.0050%  (1/20000 sampled docs)
    tag   2957:  0.0050%  (1/20000 sampled docs)
    tag   2958:  0.0050%  (1/20000 sampled docs)
    tag   2960:  0.0050%  (1/20000 sampled docs)
    tag   2962:  0.0050%  (1/20000 sampled docs)
    tag   2964:  0.0050%  (1/20000 sampled docs)
    tag   2966:  0.0050%  (1/20000 sampled docs)
    tag   2970:  0.0050%  (1/20000 sampled docs)
    tag   2975:  0.0050%  (1/20000 sampled docs)
    tag   2976:  0.0050%  (1/20000 sampled docs)
    tag   2993:  0.0050%  (1/20000 sampled docs)
    tag   3010:  0.0050%  (1/20000 sampled docs)
    tag   3016:  0.0050%  (1/20000 sampled docs)
    tag   3018:  0.0050%  (1/20000 sampled docs)
    tag   3029:  0.0050%  (1/20000 sampled docs)
    tag   3030:  0.0050%  (1/20000 sampled docs)
    tag   3042:  0.0050%  (1/20000 sampled docs)
    tag   3043:  0.0050%  (1/20000 sampled docs)
    tag   3044:  0.0050%  (1/20000 sampled docs)
    tag   3045:  0.0050%  (1/20000 sampled docs)
    tag   3046:  0.0050%  (1/20000 sampled docs)
    tag   3060:  0.0050%  (1/20000 sampled docs)
    tag   3062:  0.0050%  (1/20000 sampled docs)
    tag   3077:  0.0050%  (1/20000 sampled docs)
    tag   3085:  0.0050%  (1/20000 sampled docs)
    tag   3086:  0.0050%  (1/20000 sampled docs)
    tag   3087:  0.0050%  (1/20000 sampled docs)
    tag   3088:  0.0050%  (1/20000 sampled docs)
    tag   3090:  0.0050%  (1/20000 sampled docs)
    tag   3091:  0.0050%  (1/20000 sampled docs)
    tag   3098:  0.0050%  (1/20000 sampled docs)
    tag   3099:  0.0050%  (1/20000 sampled docs)
    tag   3114:  0.0050%  (1/20000 sampled docs)
    tag   3115:  0.0050%  (1/20000 sampled docs)
    tag   3128:  0.0050%  (1/20000 sampled docs)
    tag   3132:  0.0050%  (1/20000 sampled docs)
    tag   3133:  0.0050%  (1/20000 sampled docs)
    tag   3137:  0.0050%  (1/20000 sampled docs)
    tag   3143:  0.0050%  (1/20000 sampled docs)
    tag   3161:  0.0050%  (1/20000 sampled docs)
    tag   3175:  0.0050%  (1/20000 sampled docs)
    tag   3176:  0.0050%  (1/20000 sampled docs)
    tag   3177:  0.0050%  (1/20000 sampled docs)
    tag   3178:  0.0050%  (1/20000 sampled docs)
    tag   3180:  0.0050%  (1/20000 sampled docs)
    tag   3184:  0.0050%  (1/20000 sampled docs)
    tag   3186:  0.0050%  (1/20000 sampled docs)
    tag   3187:  0.0050%  (1/20000 sampled docs)
    tag   3190:  0.0050%  (1/20000 sampled docs)
    tag   3198:  0.0050%  (1/20000 sampled docs)
    tag   3201:  0.0050%  (1/20000 sampled docs)
    tag   3205:  0.0050%  (1/20000 sampled docs)
    tag   3208:  0.0050%  (1/20000 sampled docs)
    tag   3212:  0.0050%  (1/20000 sampled docs)
    tag   3215:  0.0050%  (1/20000 sampled docs)
    tag   3217:  0.0050%  (1/20000 sampled docs)
    tag   3219:  0.0050%  (1/20000 sampled docs)
    tag   3224:  0.0050%  (1/20000 sampled docs)
    tag   3225:  0.0050%  (1/20000 sampled docs)
    tag   3226:  0.0050%  (1/20000 sampled docs)
    tag   3227:  0.0050%  (1/20000 sampled docs)
    tag   3237:  0.0050%  (1/20000 sampled docs)
    tag   3238:  0.0050%  (1/20000 sampled docs)
    tag   3239:  0.0050%  (1/20000 sampled docs)
    tag   3240:  0.0050%  (1/20000 sampled docs)
    tag   3241:  0.0050%  (1/20000 sampled docs)
    tag   3242:  0.0050%  (1/20000 sampled docs)
    tag   3243:  0.0050%  (1/20000 sampled docs)
    tag   3244:  0.0050%  (1/20000 sampled docs)
    tag   3273:  0.0050%  (1/20000 sampled docs)
    tag   3275:  0.0050%  (1/20000 sampled docs)
    tag   3566:  0.0050%  (1/20000 sampled docs)
    tag   3569:  0.0050%  (1/20000 sampled docs)
    tag   3574:  0.0050%  (1/20000 sampled docs)
    tag   3577:  0.0050%  (1/20000 sampled docs)
    tag   3583:  0.0050%  (1/20000 sampled docs)
    tag   3584:  0.0050%  (1/20000 sampled docs)
    tag   3592:  0.0050%  (1/20000 sampled docs)
    tag   3594:  0.0050%  (1/20000 sampled docs)
    tag   3603:  0.0050%  (1/20000 sampled docs)
    tag   3604:  0.0050%  (1/20000 sampled docs)
    tag   3607:  0.0050%  (1/20000 sampled docs)
    tag   3612:  0.0050%  (1/20000 sampled docs)
    tag   3613:  0.0050%  (1/20000 sampled docs)
    tag   3617:  0.0050%  (1/20000 sampled docs)
    tag   3618:  0.0050%  (1/20000 sampled docs)
    tag   3622:  0.0050%  (1/20000 sampled docs)
    tag   3626:  0.0050%  (1/20000 sampled docs)
    tag   3627:  0.0050%  (1/20000 sampled docs)
    tag   3631:  0.0050%  (1/20000 sampled docs)
    tag   3637:  0.0050%  (1/20000 sampled docs)
    tag   3638:  0.0050%  (1/20000 sampled docs)
    tag   3644:  0.0050%  (1/20000 sampled docs)
    tag   3646:  0.0050%  (1/20000 sampled docs)
    tag   3648:  0.0050%  (1/20000 sampled docs)
    tag   3652:  0.0050%  (1/20000 sampled docs)
    tag   3653:  0.0050%  (1/20000 sampled docs)
    tag   3654:  0.0050%  (1/20000 sampled docs)
    tag   3666:  0.0050%  (1/20000 sampled docs)
    tag   3675:  0.0050%  (1/20000 sampled docs)
    tag   3676:  0.0050%  (1/20000 sampled docs)
    tag   3678:  0.0050%  (1/20000 sampled docs)
    tag   3686:  0.0050%  (1/20000 sampled docs)
    tag   3687:  0.0050%  (1/20000 sampled docs)
    tag   3692:  0.0050%  (1/20000 sampled docs)
    tag   3693:  0.0050%  (1/20000 sampled docs)
    tag   3694:  0.0050%  (1/20000 sampled docs)
    tag   3695:  0.0050%  (1/20000 sampled docs)
    tag   3697:  0.0050%  (1/20000 sampled docs)
    tag   3701:  0.0050%  (1/20000 sampled docs)
    tag   3702:  0.0050%  (1/20000 sampled docs)
    tag   3703:  0.0050%  (1/20000 sampled docs)
    tag   3704:  0.0050%  (1/20000 sampled docs)
    tag   3705:  0.0050%  (1/20000 sampled docs)
    tag   3707:  0.0050%  (1/20000 sampled docs)
    tag   3712:  0.0050%  (1/20000 sampled docs)
    tag   3713:  0.0050%  (1/20000 sampled docs)
    tag   3714:  0.0050%  (1/20000 sampled docs)
    tag   3716:  0.0050%  (1/20000 sampled docs)
    tag   3728:  0.0050%  (1/20000 sampled docs)
    tag   3729:  0.0050%  (1/20000 sampled docs)
    tag   3730:  0.0050%  (1/20000 sampled docs)
    tag   3731:  0.0050%  (1/20000 sampled docs)
    tag   3733:  0.0050%  (1/20000 sampled docs)
    tag   3742:  0.0050%  (1/20000 sampled docs)
    tag   3743:  0.0050%  (1/20000 sampled docs)
    tag   3754:  0.0050%  (1/20000 sampled docs)
    tag   3755:  0.0050%  (1/20000 sampled docs)
    tag   3761:  0.0050%  (1/20000 sampled docs)
    tag   3764:  0.0050%  (1/20000 sampled docs)
    tag   3765:  0.0050%  (1/20000 sampled docs)
    tag   3768:  0.0050%  (1/20000 sampled docs)
    tag   3769:  0.0050%  (1/20000 sampled docs)
    tag   3772:  0.0050%  (1/20000 sampled docs)
    tag   3776:  0.0050%  (1/20000 sampled docs)
    tag   3777:  0.0050%  (1/20000 sampled docs)
    tag   3778:  0.0050%  (1/20000 sampled docs)
    tag   3779:  0.0050%  (1/20000 sampled docs)
    tag   3782:  0.0050%  (1/20000 sampled docs)
    tag   3783:  0.0050%  (1/20000 sampled docs)
    tag   3786:  0.0050%  (1/20000 sampled docs)
    tag   3794:  0.0050%  (1/20000 sampled docs)
    tag   3813:  0.0050%  (1/20000 sampled docs)
    tag   3823:  0.0050%  (1/20000 sampled docs)
    tag   3824:  0.0050%  (1/20000 sampled docs)
    tag   3825:  0.0050%  (1/20000 sampled docs)
    tag   3837:  0.0050%  (1/20000 sampled docs)
    tag   3838:  0.0050%  (1/20000 sampled docs)
    tag   3840:  0.0050%  (1/20000 sampled docs)
    tag   3845:  0.0050%  (1/20000 sampled docs)
    tag   3846:  0.0050%  (1/20000 sampled docs)
    tag   3874:  0.0050%  (1/20000 sampled docs)
    tag   3898:  0.0050%  (1/20000 sampled docs)
    tag   3899:  0.0050%  (1/20000 sampled docs)
    tag   3900:  0.0050%  (1/20000 sampled docs)
    tag   3901:  0.0050%  (1/20000 sampled docs)
    tag   3902:  0.0050%  (1/20000 sampled docs)
    tag   3903:  0.0050%  (1/20000 sampled docs)
    tag   3904:  0.0050%  (1/20000 sampled docs)
    tag   3905:  0.0050%  (1/20000 sampled docs)
    tag   3906:  0.0050%  (1/20000 sampled docs)
    tag   3908:  0.0050%  (1/20000 sampled docs)
    tag   3911:  0.0050%  (1/20000 sampled docs)
    tag   3912:  0.0050%  (1/20000 sampled docs)
    tag   3913:  0.0050%  (1/20000 sampled docs)
    tag   3914:  0.0050%  (1/20000 sampled docs)
    tag   3915:  0.0050%  (1/20000 sampled docs)
    tag   3935:  0.0050%  (1/20000 sampled docs)
    tag   3936:  0.0050%  (1/20000 sampled docs)
    tag   3937:  0.0050%  (1/20000 sampled docs)
    tag   3938:  0.0050%  (1/20000 sampled docs)
    tag   3939:  0.0050%  (1/20000 sampled docs)
    tag   3940:  0.0050%  (1/20000 sampled docs)
    tag   3941:  0.0050%  (1/20000 sampled docs)
    tag   3942:  0.0050%  (1/20000 sampled docs)
    tag   3943:  0.0050%  (1/20000 sampled docs)
    tag   3944:  0.0050%  (1/20000 sampled docs)
    tag   3947:  0.0050%  (1/20000 sampled docs)
    tag   3956:  0.0050%  (1/20000 sampled docs)
    tag   3957:  0.0050%  (1/20000 sampled docs)
    tag   3958:  0.0050%  (1/20000 sampled docs)
    tag   3959:  0.0050%  (1/20000 sampled docs)
    tag   3960:  0.0050%  (1/20000 sampled docs)
    tag   3961:  0.0050%  (1/20000 sampled docs)
    tag   3962:  0.0050%  (1/20000 sampled docs)
    tag   4003:  0.0050%  (1/20000 sampled docs)
    tag   4004:  0.0050%  (1/20000 sampled docs)
    tag   4005:  0.0050%  (1/20000 sampled docs)
    tag   4006:  0.0050%  (1/20000 sampled docs)
    tag   4007:  0.0050%  (1/20000 sampled docs)
    tag   4008:  0.0050%  (1/20000 sampled docs)
    tag   4009:  0.0050%  (1/20000 sampled docs)
    tag   4010:  0.0050%  (1/20000 sampled docs)
  (+ 16990 tags NEVER seen in the sample -> effectively absent)

-- index storage implication (doc_num_per_segment=65536, density_threshold=0.05) --
  segments=153  bitset-tags(freq>=5%)=1001  bitlist/absent-tags=34671
  ~bitset posting storage = bitset_tags * segments * (seg/8) = 1.3 GB (dense common tags, present in ~all segments -> dominant term)

-- document vectors --
  dim=64  component mean=-0.08314 std=4.268  mean L2 norm=30.47  (normalized=no)
  L2 norm: n=20000 min=2.952 p50=26.15 mean=30.47 p90=53.49 p99=82.48 max=217.4
    [     2.952,     20.82)      4524 ###############
    [     20.82,     38.69)     12079 ########################################
    [     38.69,     56.56)      1625 #####
    [     56.56,     74.43)      1295 ####
    [     74.43,      92.3)       386 #
    [      92.3,     110.2)        78 
    [     110.2,       128)         5 
    [       128,     145.9)         2 
    [     145.9,     163.8)         3 
    [     163.8,     181.6)         1 
    [     181.6,     199.5)         1 
    [     199.5,     217.4)         1 
  component value range (sampled): [-83.15, 72.87]

### QUERYDATA  QueryData_10000.txt   (10908 records)
  parsed OK: 10908   unparseable json_query: 0
  with syntax_filter: 0   match-all (no filter): 10908

-- result_num (top_k) --
  result_num: n=10908 min=3500 p50=3500 mean=3500 p90=3500 p99=3500 max=3500
    all == 3500  (count 10908)

-- query vector --
  dim(s): {64: 10888}
  L2 norm: n=10888 min=4.099 p50=4.195 mean=4.329 p90=4.966 p99=4.966 max=5.695
    [     4.099,     4.232)      6724 ########################################
    [     4.232,     4.365)      1206 #######
    [     4.365,     4.498)       369 ##
    [     4.498,     4.631)       932 ######
    [     4.631,     4.764)       318 ##
    [     4.764,     4.897)        76 
    [     4.897,      5.03)      1173 #######
    [      5.03,     5.163)         0 
    [     5.163,     5.296)        25 
    [     5.296,     5.429)         0 
    [     5.429,     5.562)        64 
    [     5.562,     5.695)         1 
  component value range: [-1.747, 1.607]
