def rule_based_baseline(data):
    return (
        (data["mean_speed"] >= 2) &
        (data["mean_speed"] <= 5) &
        (data["turning_rate"] > 10)
    ).astype(int)
