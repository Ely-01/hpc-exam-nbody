| Threads | Kernel | Accumulators | Repeats | Force median s | Force stdev s | Speedup vs direct | Marginal gain | Gpair/s | Nominal GFLOP/s | Max drift | Status |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | direct | 1 | 5 | 5.821120 | 0.024255 | 1.000 | 0.00% | 0.242 | 4.84 | 4.083561e-08 | OK |
| 1 | direct-split2 | 2 | 5 | 5.659731 | 0.003013 | 1.029 | 2.85% | 0.249 | 4.98 | 4.083561e-08 | OK |
| 1 | direct-split4 | 4 | 5 | 5.697840 | 0.003039 | 1.022 | -0.67% | 0.247 | 4.95 | 4.083561e-08 | OK |
| 1 | direct-split8 | 8 | 5 | 5.656612 | 0.008691 | 1.029 | 0.73% | 0.249 | 4.98 | 4.083561e-08 | OK |
| 2 | direct | 1 | 5 | 3.094624 | 0.208808 | 1.000 | 0.00% | 0.455 | 9.11 | 4.083561e-08 | OK |
| 2 | direct-split2 | 2 | 5 | 2.879254 | 0.012164 | 1.075 | 7.48% | 0.489 | 9.79 | 4.083561e-08 | OK |
| 2 | direct-split4 | 4 | 5 | 2.857590 | 0.001465 | 1.083 | 0.76% | 0.493 | 9.86 | 4.083561e-08 | OK |
| 2 | direct-split8 | 8 | 5 | 2.894601 | 0.027718 | 1.069 | -1.28% | 0.487 | 9.74 | 4.083561e-08 | OK |
| 4 | direct | 1 | 5 | 1.743009 | 0.008529 | 1.000 | 0.00% | 0.808 | 16.17 | 4.083561e-08 | OK |
| 4 | direct-split2 | 2 | 5 | 1.595104 | 0.057314 | 1.093 | 9.27% | 0.883 | 17.67 | 4.083561e-08 | OK |
| 4 | direct-split4 | 4 | 5 | 1.594307 | 0.056764 | 1.093 | 0.05% | 0.884 | 17.68 | 4.083561e-08 | OK |
| 4 | direct-split8 | 8 | 5 | 1.592580 | 0.002604 | 1.094 | 0.11% | 0.885 | 17.70 | 4.083561e-08 | OK |
| 8 | direct | 1 | 5 | 0.898525 | 0.056340 | 1.000 | 0.00% | 1.568 | 31.37 | 4.083561e-08 | OK |
| 8 | direct-split2 | 2 | 5 | 0.805495 | 0.030483 | 1.115 | 11.55% | 1.749 | 34.99 | 4.083561e-08 | OK |
| 8 | direct-split4 | 4 | 5 | 0.796616 | 0.029899 | 1.128 | 1.11% | 1.769 | 35.38 | 4.083561e-08 | OK |
| 8 | direct-split8 | 8 | 5 | 0.780765 | 0.022961 | 1.151 | 2.03% | 1.805 | 36.10 | 4.083561e-08 | OK |

Saturation notes:
- 1 thread(s): best direct-split8 (1.029x vs direct); saturates around 2 accumulators (2.9% marginal gain).
- 2 thread(s): best direct-split4 (1.083x vs direct); saturates around 4 accumulators (0.8% marginal gain).
- 4 thread(s): best direct-split8 (1.094x vs direct); saturates around 4 accumulators (0.1% marginal gain).
- 8 thread(s): best direct-split8 (1.151x vs direct); saturates around 4 accumulators (1.1% marginal gain).

The nominal GFLOP/s column uses an approximate flop count per pair; use it as a relative kernel-throughput indicator, not as a hardware-counter replacement.
