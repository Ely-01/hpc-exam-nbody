| Metric | Message size bytes | Repeats | Native median ± sigma | Container median ± sigma | Relative difference |
|:---|---:|---:|---:|---:|---:|
| Latency (us, lower is better) | 1 | 5 | 9.940 ± 0.161 | 10.000 ± 0.088 | +0.60% |
| Bandwidth (MB/s, higher is better) | 4194304 | 5 | 1673.450 ± 245.560 | 1668.570 ± 244.222 | -0.29% |

The OSU Micro-Benchmarks were built locally in user space because they were not available as Orfeo modules. The container run uses the same host OpenMPI runtime policy as the application container runs. Positive latency difference means the container is slower; positive bandwidth difference means the container measured higher bandwidth.
