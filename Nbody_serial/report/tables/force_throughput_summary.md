| Ranks | Threads | Workers | Force median +/- sigma s | Force / total | Ginteraction/s | Ideal Ginteraction/s | Throughput efficiency | Nominal GFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 370.531173 +/- 0.489564 | 98.97% | 0.293 | 0.293 | 1.000 | 5.85 |
| 2 | 1 | 2 | 185.020960 +/- 0.084704 | 98.51% | 0.586 | 0.585 | 1.001 | 11.72 |
| 4 | 1 | 4 | 92.475055 +/- 0.019084 | 98.27% | 1.173 | 1.171 | 1.002 | 23.45 |
| 8 | 1 | 8 | 46.242101 +/- 0.003023 | 98.08% | 2.345 | 2.341 | 1.002 | 46.90 |
| 16 | 1 | 16 | 23.142003 +/- 0.001770 | 97.98% | 4.686 | 4.683 | 1.001 | 93.72 |
| 32 | 1 | 32 | 11.584479 +/- 0.006580 | 97.59% | 9.361 | 9.366 | 1.000 | 187.22 |

Throughput uses ordered source-target particle interactions:

`N * (N - 1) * (nsteps + 1) / force_median_s`.

The nominal GFLOP/s column assumes 20 floating-point operations per interaction and is a relative throughput indicator, not a hardware-counter measurement.
