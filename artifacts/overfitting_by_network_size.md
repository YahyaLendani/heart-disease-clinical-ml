# Overfitting Analysis by Network Size

Sorted by generalization gap (Best Val Acc - Test Acc, higher usually indicates stronger overfitting).

| Model | Hidden Dims | Best Val Acc | Test Acc | Gen Gap (Val-Test) | Best Loss Gap | Final Loss Gap | CV Acc Mean ± Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 64,32 | 0.7551 | 0.7213 | 0.0338 | 0.1151 | 0.3724 | 0.8258 ± 0.0093 |
| large | 128,64 | 0.7347 | 0.7377 | -0.0030 | 0.7303 | 1.1138 | 0.8464 ± 0.0162 |
| small | 32,16 | 0.6531 | 0.7541 | -0.1010 | 0.2959 | 0.2731 | 0.8131 ± 0.0535 |