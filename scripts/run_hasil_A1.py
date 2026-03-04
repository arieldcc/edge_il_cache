import json

log = json.load(open("results/wikipedia_september_2007/001_ilnse_A1_12221.jsonl"))  # sesuaikan loader-mu
for s in log["slots"]:
    mean = s["feature_stats"]["overall"]["mean"][-1]
    std  = s["feature_stats"]["overall"]["std"][-1]
    mn   = s["feature_stats"]["overall"]["min"][-1]
    mx   = s["feature_stats"]["overall"]["max"][-1]
    if mx > mn:  # ada variasi
        print("VARIATIVE at slot", s["slot_index"], mn, mx, std)
        break
else:
    print("slot_freq feature looks degenerate/constant across all slots")
