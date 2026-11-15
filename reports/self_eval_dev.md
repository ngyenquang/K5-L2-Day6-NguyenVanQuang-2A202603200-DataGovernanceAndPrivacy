# Self-evaluation on dev (own labels, NOT the official metric)

dev N=56, defects=28. Seed 42. Pool = dev images only.

| Run | B | Recall@B | Precision@B | hits | type acc |
|---|---|---|---|---|---|
| baseline (Laplacian) | 20 | 0.571 | 0.800 | 16/28 | n/a |
| PoC | 20 | 0.714 | 1.000 | 20/28 | 1.00 |
| baseline (Laplacian) | 28 | 0.821 | 0.821 | 23/28 | n/a |
| PoC | 28 | 1.000 | 1.000 | 28/28 | 1.00 |

Note: B=20 caps recall below 1.0 on dev because dev has more than 20 defects; B=#defects is the ceiling case.

## False alarms by slice (B=20)
- baseline (Laplacian): false alarms in top-20 (clean images in top-20 / clean images) -> light=day: 3/18, light=dusk: 1/8, light=night: 0/2, camera_id=cam_A: 0/18, camera_id=cam_B: 4/10
- PoC: false alarms in top-20 (clean images in top-20 / clean images) -> light=day: 0/18, light=dusk: 0/8, light=night: 0/2, camera_id=cam_A: 0/18, camera_id=cam_B: 0/10

Caveat: PoC thresholds were chosen on these same dev images (optimistic); baseline has no tuned parameters.
