# Overfitting Analysis by Network Size

Sorted by generalization gap (Best Val Acc - Test Acc, higher usually indicates stronger overfitting).

| Model | Hidden Dims | Best Val Acc | Test Acc | Gen Gap (Val-Test) | Best Loss Gap | Final Loss Gap | CV Acc Mean ± Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 32,16 | 0.7959 | 0.7869 | 0.0090 | 0.0618 | 0.0977 | 0.7939 ± 0.0418 |
| base | 64,32 | 0.7755 | 0.8197 | -0.0442 | 0.1711 | 0.1711 | 0.7567 ± 0.0879 |
| large | 128,64 | 0.7755 | 0.8689 | -0.0933 | 0.1195 | 0.3336 | 0.7677 ± 0.0399 |