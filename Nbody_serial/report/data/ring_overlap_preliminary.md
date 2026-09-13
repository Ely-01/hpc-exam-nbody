| N | Ranks | Threads | Repeats | Blocking total s | Overlap total s | Blocking comm s | Exposed overlap comm s | Hidden comm | Overlap achieved | Runtime saved | Napkin hidden | Speedup | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 8192 | 8 | 1 | 5 | 0.581731 | 0.579076 | 0.059508 | 0.055438 | 0.004070 | 6.8% | 0.002655 | 100.0% | 1.005 | OK/OK |

Interpretation:
- `Hidden comm` is the reduction in exposed communication time: blocking communication median minus overlap-mode MPI post/wait median.
- `Overlap achieved` is `hidden_comm / blocking_comm`; 100% would mean the communication cost is fully hidden by useful force work.
- `Napkin hidden` is the optimistic bound from comparing blocking communication with force work. Real measurements are usually lower because MPI progress may require entering MPI, messages have startup/injection costs, OpenMP threads and MPI progress can compete for cores, and phase imbalance leaves some ranks waiting.
- `Runtime saved` can be smaller than hidden communication because total time also includes energy diagnostics, reductions, scheduling noise, and any overhead introduced by non-blocking calls.
