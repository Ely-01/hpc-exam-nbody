| Mode | Repeats | Median s | Stdev s | Min s | Max s | Overhead vs native s |
|:---|---:|---:|---:|---:|---:|---:|
| container_true | 10 | 0.102052 | 0.196828 | 0.100407 | 0.724725 | 0.101609 |
| native_true | 10 | 0.000444 | 0.000120 | 0.000432 | 0.000823 | 0.000000 |

The container launch overhead is the fixed one-process startup cost measured with `singularity exec image true`; it is reported separately from long N-body timings because it is amortized by production runs.
