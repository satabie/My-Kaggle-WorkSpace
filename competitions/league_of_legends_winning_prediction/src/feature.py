import numpy as np


def create_features(df):
    df_new = df.copy()

    # ==KDA関連==
    df_new["kda"] = (df["kills"] + df["assists"] + 1) / np.maximum(1, df["deaths"])
    df_new["pure_kill_ratio"] = df["kills"] / np.maximum(1, df["deaths"])

    # ==ダメージ関連==
    df_new["dmg_per_death"] = df["totdmgtochamp"] / np.maximum(1, df["deaths"])
    df_new["dmg_efficiency"] = df["totdmgtochamp"] / np.maximum(1, df["totdmgtaken"])
    df["chmap_dmg_ratio"] = df["totdmgtochamp"] / np.maximum(1, df["totdmgdealt"])

    # ==ダメージ構成==
    df_new["physical_ratio"] = df["physicaldmgdealt"] / np.maximum(1, df["totdmgdealt"])
    df_new["magic_ratio"] = df["magicdmgdealt"] / np.maximum(1, df["totdmgdealt"])

    # ==生存力==
    df_new["survival_rate"] = df["longesttimespentliving"] / np.maximum(1, df["deaths"])
    df_new["tankiness"] = df["totdmgtaken"] / np.maximum(1, df["deaths"])

    # ==回復==
    df_new["healing_efficiency"] = df["totheal"] / np.maximum(1, df["totdmgtaken"])

    # === オブジェクト貢献 ===
    df_new["tower_dmg_normalized"] = df["dmgtoturrets"] / np.maximum(1, df["longesttimespentliving"])
    df_new["vision_score"] = df["wardsplaced"] + df["wardskilled"] * 2

    # === キャリー能力 ===
    df_new["multiltikill_score"] = (
        df["doublekills"] * 2 + df["triplekills"] * 4 + df["quadrakills"] * 8 + df["pentakills"] * 16
    )

    # === 総合指標 ===
    df_new["overall_performance"] = (
        df_new["kda"] * 0.6
        + np.log1p(df_new["dmgtoturrets"]) * 0.2
        + df_new["largestkillingspree"] * 0.2
        + df_new["multiltikill_score"] * 0.3
    )

    return df_new
