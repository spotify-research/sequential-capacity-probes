# Table 1 results

| Data | MC | FMC | FMC+ | SAS+ | eSAS | SeqRules | PCTM |
|---|---:|---:|---:|---:|---:|---:|---:|
| beauty | 0.0492 | 0.0481 | 0.0531 | 0.0534 | 0.0511 | 0.0605 | 0.0635 |
| sports | 0.0251 | 0.0244 | 0.0271 | 0.0321 | 0.0320 | 0.0371 | 0.0368 |
| toys | 0.0548 | 0.0559 | 0.0588 | 0.0587 | 0.0516 | 0.0730 | 0.0738 |
| ml1m | 0.1240 | 0.1124 | 0.1206 | 0.1654 | 0.1767 | 0.1505 | 0.1815 |
| ml20m | 0.1050 | 0.0807 | 0.1034 | 0.1814 | 0.1948 | 0.1115 | 0.1431 |

Status: PASS

MC, SeqRules, and PCTM must match the reported four-decimal value. 
FMC and FMC+ must be within absolute NDCG@10 tolerance 0.001; 
SASRec+ and eSASRec must be within tolerance 0.005.
