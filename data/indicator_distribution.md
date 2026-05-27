# Bias Sampling — Before & After Proportions

Columns show **count (%)** for each scenario vs original dataset.
- **balanced**: equal count per indicator
- **imbalanced**: real data → proportional to actual pool size (mirrors original); synthetic → Zipf by pool size desc (artificial imbalance, since original is already balanced)
- **rare_heavy**: Zipf weights by pool size asc (rare themes boosted)

## ESConv (problem_type, filtered ≥50) — real data

| Indicator | original | balanced | imbalanced | rare_heavy |
|---|---:|---:|---:|---:|
| ongoing depression | 351 (29.1%) | 156 (20.0%) | 351 (29.1%) | 31 (8.7%) |
| job crisis | 280 (23.2%) | 156 (20.0%) | 280 (23.2%) | 39 (11.0%) |
| breakup with partner | 239 (19.8%) | 156 (20.0%) | 239 (19.8%) | 52 (14.6%) |
| problems with friends | 179 (14.9%) | 156 (20.0%) | 179 (14.9%) | 78 (21.9%) |
| academic pressure | 156 (12.9%) | 156 (20.0%) | 156 (12.9%) | 156 (43.8%) |
| **TOTAL** | **1205 (100%)** | **780 (100%)** | **1205 (100%)** | **356 (100%)** |

## AI Healthcare — synthetic data

| Indicator | original | balanced | imbalanced | rare_heavy |
|---|---:|---:|---:|---:|
| System improvements | 72 (14.4%) | 71 (14.3%) | 24 (12.9%) | 14 (7.6%) |
| Improve quality of patient care and shared decision-making | 72 (14.4%) | 71 (14.3%) | 72 (38.7%) | 10 (5.4%) |
| Challenges with security, bias, and access | 72 (14.4%) | 71 (14.3%) | 36 (19.4%) | 12 (6.5%) |
| Acknowledging and supporting diversity in public involvement | 71 (14.2%) | 71 (14.3%) | 12 (6.5%) | 35 (19.0%) |
| Experience, empowerment, and raising awareness | 71 (14.2%) | 71 (14.3%) | 14 (7.5%) | 24 (13.0%) |
| Public misunderstanding of AI | 71 (14.2%) | 71 (14.3%) | 18 (9.7%) | 18 (9.8%) |
| Lack of human touch in care and decision-making | 71 (14.2%) | 71 (14.3%) | 10 (5.4%) | 71 (38.6%) |
| **TOTAL** | **500 (100%)** | **497 (100%)** | **186 (100%)** | **184 (100%)** |

## Climate (multi-label pools) — real data

| Indicator | original | balanced | imbalanced | rare_heavy |
|---|---:|---:|---:|---:|
| PA | 384 (29.2%) | 47 (14.3%) | 384 (29.2%) | 7 (5.7%) |
| CA | 300 (22.8%) | 47 (14.3%) | 300 (22.8%) | 8 (6.6%) |
| CB | 226 (17.2%) | 47 (14.3%) | 226 (17.2%) | 9 (7.4%) |
| GA | 141 (10.7%) | 47 (14.3%) | 141 (10.7%) | 12 (9.8%) |
| GC | 115 (8.8%) | 47 (14.3%) | 115 (8.8%) | 16 (13.1%) |
| SA | 100 (7.6%) | 47 (14.3%) | 100 (7.6%) | 23 (18.9%) |
| PB | 47 (3.6%) | 47 (14.3%) | 47 (3.6%) | 47 (38.5%) |
| **TOTAL** | **1313 (100%)** | **329 (100%)** | **1313 (100%)** | **122 (100%)** |

## School Burnout — synthetic data

| Indicator | original | balanced | imbalanced | rare_heavy |
|---|---:|---:|---:|---:|
| low motivation / thoughts of giving up | 50 (11.1%) | 50 (11.1%) | 10 (7.1%) | 10 (7.1%) |
| school pressure harming close relationships | 50 (11.1%) | 50 (11.1%) | 12 (8.5%) | 8 (5.7%) |
| poor sleep due to schoolwork | 50 (11.1%) | 50 (11.1%) | 25 (17.7%) | 6 (4.3%) |
| brooding over schoolwork during free time | 50 (11.1%) | 50 (11.1%) | 17 (12.1%) | 7 (5.0%) |
| loss of interest | 50 (11.1%) | 50 (11.1%) | 8 (5.7%) | 12 (8.5%) |
| lower expectations for one's schoolwork than before | 50 (11.1%) | 50 (11.1%) | 6 (4.3%) | 50 (35.5%) |
| overwhelmed by schoolwork | 50 (11.1%) | 50 (11.1%) | 50 (35.5%) | 6 (4.3%) |
| feeling inadequate in schoolwork | 50 (11.1%) | 50 (11.1%) | 6 (4.3%) | 25 (17.7%) |
| questioning the meaning of schoolwork | 50 (11.1%) | 50 (11.1%) | 7 (5.0%) | 17 (12.1%) |
| **TOTAL** | **450 (100%)** | **450 (100%)** | **141 (100%)** | **141 (100%)** |
