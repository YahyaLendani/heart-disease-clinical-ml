# Overfitting Analysis by Network Size

Sorted by generalization gap (Best Val Acc - Test Acc, higher usually indicates stronger overfitting).

| Model | Hidden Dims | Best Val Acc | Test Acc | Gen Gap (Val-Test) | Best Loss Gap | Final Loss Gap | CV Acc Mean ± Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| large | 128,64 | 0.8027 | 0.7880 | 0.0147 | 0.1086 | 0.3917 | 0.7816 ± 0.0491 |
| small | 32,16 | 0.8027 | 0.7989 | 0.0038 | -0.0435 | 0.0110 | 0.7866 ± 0.0340 |
| base | 64,32 | 0.8095 | 0.8098 | -0.0003 | 0.0410 | 0.0201 | 0.7893 ± 0.0304 |