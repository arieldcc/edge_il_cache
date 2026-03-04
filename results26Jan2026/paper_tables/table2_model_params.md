| Symbol | Description | Value |
| --- | --- | --- |
| L | Number of gap/recency features | 6 |
| ρ | Top-K labeling ratio (K = floor(ρ·N_t)) | 0.2 |
| a | Learn++.NSE sigmoid slope | 0.5 |
| b | Learn++.NSE sigmoid shift | 10 |
| H_max | Max learners (prune oldest) | 20 |
| g_miss | Missing-gap padding value | 1e+06 |
| w_short | Short frequency window (slots) | 1 |
| w_mid | Mid frequency window (slots) | 7 |
| w_long | Long frequency window (slots) | 30 |
| w_cum | Cumulative frequency window | all past slots |
| C (%) | Cache sizes as % of distinct objects | 0.8, 1, 2, 3, 4, 5 |
| α_0 | Base admission rate | 0.08 |
| α_min | Min admission rate | 0.005 |
| α_max | Max admission rate | 0.2 |
| β_d | Drift EMA (d̄_t) | 0.3 |
| β_stats | Drift mean/var EMA | 0.1 |
| w_JSD | Drift weight for JSD | 0.4 |
| w_OV | Drift weight for overlap | 0.6 |
| g_d | Drift gain | 1 |
| α_floor | Alpha floor multiplier | 0.7 |
| γ_miss | Miss-rate pressure coefficient | 0.5 |
| q | Score-spread quantile | 0.9 |
| β_s | Score-spread EMA | 0.2 |
| q_min | Min quality multiplier | 0.7 |
| q_max | Max quality multiplier | 1.3 |
| δ_q | Quality min-boost (capacity effect) | 0.1 |
| p_target | Admission precision target | 0.12 |
| κ_p | Precision sensitivity | 1 |
| κ_c,min | Min capacity scale | 0.4 |
| ϕ_gate | Score-gate top percentile | 0.05 |
| ϕ_fill | Fill-phase ratio | 0.9 |
| r_fill | Fill-phase minimum rate | 0.05 |
| W_poll | Pollution window (slots) | 1 |
| Base learner | Classifier used in ensemble | GaussianNaiveBayes |
| Feature set (gap-only) | x = [gap1..gapL], dim L (A0 ablation) | 6 |
| Feature set (gap + multi-timescale freq) | x = [gap1..gapL, f_short, f_mid, f_long, f_cum], dim L+4 (A2 ablation) | 10 |
