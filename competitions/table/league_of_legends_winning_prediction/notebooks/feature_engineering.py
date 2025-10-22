import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, RobustScaler
    import warnings

    warnings.filterwarnings("ignore")
    return pd, plt


@app.cell
def _(mo):
    mo.md(r"# 前処理")
    return


@app.cell
def _(pd):
    train_df = pd.read_csv("./data/raw/train.csv")
    test_df = pd.read_csv("./data/raw/test_template.csv")
    return test_df, train_df


@app.cell
def _(test_df, train_df):
    # 元のデータサイズを保持
    original_train_size = len(train_df)
    original_test_size = len(test_df)

    print(original_train_size, original_test_size)
    return


@app.cell
def _(test_df, train_df):
    train_df.drop("timecc", axis=1, inplace=True)
    test_df.drop("timecc", axis=1, inplace=True)
    train_df.columns
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 外れ値検出
        今回はゲームデータなので、外れ値を除去しない方が良いと判断。
        Zスコアというものを使うらしい。
    
        $$
        z = \frac{(X - \mu)}{\sigma}
        $$
    
        標準化の式と同じだが、Zスコアは各データが平均からどれだけ離れているかを示す指標である。
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"# 特徴量エンジニアリング")
    return


@app.cell
def _():
    import sys
    sys.path.append("./src")
    from feature import create_features
    return (create_features,)


@app.cell
def _(create_features, plt, test_df, train_df):
    train_df_add_features = create_features(train_df)
    test_df_add_features = create_features(test_df)

    # 特徴量とターゲットの分割
    correlations = train_df_add_features.corrwith(train_df_add_features["win"]).sort_values(ascending=False)

    print("勝敗と最も相関が高い特徴量 TOP 10:")
    print(correlations.head(10))

    print("勝敗と最も相関が低い特徴量 TOP 10:")
    print(correlations.tail(10))

    # 可視化
    plt.figure(figsize=(10, 8))
    correlations.plot(kind="barh", color=correlations.apply(lambda x: "green" if x > 0 else "red"))
    plt.title("Correlations with Win", fontsize=14, fontweight="bold")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.show()
    return test_df_add_features, train_df_add_features


@app.cell
def _(test_df_add_features, train_df_add_features):
    # データの分割とスケーリン
    train_x_final = train_df_add_features.drop(["id"], axis=1)
    test_x_final = test_df_add_features

    # 行数の確認
    print(train_x_final.shape, test_x_final.shape)
    return test_x_final, train_x_final


@app.cell
def _(test_x_final, train_x_final):
    # それぞれをcsvで保存
    save_dir = "./data/processed/"
    # ディレクトリが存在しない場合は作成
    import os

    os.makedirs(save_dir, exist_ok=True)
    train_x_final.to_csv(save_dir + "train_x_add_features_2.csv", index=False)
    test_x_final.to_csv(save_dir + "test_x_add_features_2.csv", index=False)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
