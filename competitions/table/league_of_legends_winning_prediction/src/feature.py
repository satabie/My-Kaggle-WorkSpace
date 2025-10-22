import numpy as np


def create_features(df):
    df_new = df.copy()

    # ==KDA関連==
    df_new["kda"] = (df["kills"] + df["assists"] + 1) / np.maximum(1, df["deaths"])
    df_new["pure_kill_ratio"] = df["kills"] / np.maximum(1, df["deaths"])

    # ==ダメージ関連==
    df_new["dmg_per_death"] = df["totdmgtochamp"] / np.maximum(1, df["deaths"])
    df_new["dmg_efficiency"] = df["totdmgtochamp"] / np.maximum(1, df["totdmgtaken"])
    df_new["chmap_dmg_ratio"] = df["totdmgtochamp"] / np.maximum(1, df["totdmgdealt"])

    # ==ダメージ構成==
    df_new["physical_ratio"] = df["physicaldmgdealt"] / np.maximum(1, df["totdmgdealt"])
    df_new["magic_ratio"] = df["magicdmgdealt"] / np.maximum(1, df["totdmgdealt"])

    df_new["crit_to_physical_ratio"] = df["largestcrit"] / np.maximum(1, df["physicaldmgdealt"])
    df_new["magic_dmg_taken_ratio"] = df["magicdmgtaken"] / np.maximum(1, df["totdmgtaken"])
    df_new["true_dmg_taken_ratio"] = df["truedmgtaken"] / np.maximum(1, df["totdmgtaken"])

    # ==生存力==
    df_new["survival_rate"] = df["longesttimespentliving"] / np.maximum(1, df["deaths"])
    df_new["tankiness"] = df["totdmgtaken"] / np.maximum(1, df["deaths"])

    df_new["true_dmg_ratio"] = df["truedmgtochamp"] / np.maximum(1, df["totdmgtochamp"])

    # ==回復==
    df_new["healing_efficiency"] = df["totheal"] / np.maximum(1, df["totdmgtaken"])

    # === オブジェクト貢献 ===
    df_new["tower_dmg_normalized"] = df["dmgtoturrets"] / np.maximum(1, df["longesttimespentliving"])
    df_new["vision_score"] = df["wardsplaced"] + df["wardskilled"] * 2

    # === キャリー能力 ===
    df_new["weighted_multikill"] = (
        df["largestkillingspree"] * df["largestmultikill"]
        + df["doublekills"] * 2
        + df["triplekills"] * 5
        + df["quadrakills"] * 12
        + df["pentakills"] * 30
    ) / np.maximum(
        1, df["longesttimespentliving"] / 60
    )  # 1分あたりに正規化

    # === ビジョンと戦闘のバランス ===
    df_new["vision_combat_balance"] = (
        (df["wardsplaced"] + df["wardskilled"] * 2) * np.sqrt(df["totdmgtochamp"])
    ) / np.maximum(1, df["deaths"] + 1)

    # === 総合指標 ===
    df_new["overall_performance"] = (
        df_new["kda"] * 0.6
        + np.log1p(df_new["dmgtoturrets"]) * 0.2
        + df_new["largestkillingspree"] * 0.2
        + df_new["weighted_multikill"] * 0.3
    )

    return df_new
