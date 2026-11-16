# Safety Shield Ablation Report

## Method

The same 100 deterministic command scenarios were evaluated with the raw
velocity command (shield off) and with `SafetyShield.filter_velocity` (shield
on). Each scenario contains a static obstacle and a nearby drone. Collision
checks use continuous swept segments rather than discrete waypoint samples.

Run the experiment with:

```text
python safety_ablation.py
```

## Result

| Metric | Shield off | Shield on |
|---|---:|---:|
| Scenarios | 100 | 100 |
| Collisions | 100 | 0 |
| Near misses | 0 | 0 |

The shield changes unsafe commands to deterministic safe alternatives and
prevents the high-speed obstacle tunnelling demonstrated by the raw commands.
