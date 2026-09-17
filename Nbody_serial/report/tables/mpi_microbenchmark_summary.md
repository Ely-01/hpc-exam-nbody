| Metric | Message size bytes | Repeats | Native median ± sigma | Container median ± sigma | Relative difference |
|:---|---:|---:|---:|---:|---:|
| Latency (us, lower is better) | 1 | 5 | 14.780 ± 0.267 | 15.230 ± 0.402 | +3.04% |
| Bandwidth (MB/s, higher is better) | 4194304 | 5 | 1600.150 ± 83.527 | 1577.820 ± 11.696 | -1.40% |

The OSU Micro-Benchmarks were built locally in user space because they were not available as Orfeo modules. The container run uses the same host OpenMPI runtime policy as the application container runs. Positive latency difference means the container is slower; positive bandwidth difference means the container measured higher bandwidth.
