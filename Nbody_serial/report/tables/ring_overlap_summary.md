| N | Ranks | Threads | Repeats | Blocking total s | Overlap total s | Blocking comm s | Exposed overlap comm s | Hidden comm | Overlap achieved | Runtime saved | Napkin hidden | Speedup | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 32768 | 4 | 1 | 5 | 20.987808 | 20.868140 | 1.036399 | 0.866720 | 0.169679 | 16.4% | 0.119668 | 100.0% | 1.006 | OK/OK |
| 32768 | 8 | 1 | 5 | 10.513724 | 10.510297 | 0.678918 | 0.658784 | 0.020134 | 3.0% | 0.003427 | 100.0% | 1.000 | OK/OK |
| 32768 | 16 | 1 | 5 | 5.358677 | 5.361461 | 0.478483 | 0.479699 | 0.000000 | 0.0% | -0.002784 | 100.0% | 0.999 | OK/OK |
| 32768 | 32 | 1 | 5 | 2.729329 | 2.716461 | 0.226680 | 0.211975 | 0.014705 | 6.5% | 0.012868 | 100.0% | 1.005 | OK/OK |

Interpretation:
- `Hidden comm` is the reduction in exposed communication time: blocking communication median minus overlap-mode MPI post/wait median.
- `Overlap achieved` is `hidden_comm / blocking_comm`; 100% would mean the communication cost is fully hidden by useful force work.
- `Napkin hidden` is the optimistic bound from comparing blocking communication with force work. Real measurements are usually lower because MPI progress may require entering MPI, messages have startup/injection costs, OpenMP threads and MPI progress can compete for cores, and phase imbalance leaves some ranks waiting.
- `Runtime saved` can be smaller than hidden communication because total time also includes energy diagnostics, reductions, scheduling noise, and any overhead introduced by non-blocking calls.
