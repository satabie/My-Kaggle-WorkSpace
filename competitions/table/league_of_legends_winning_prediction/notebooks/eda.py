import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    return np, pd, plt


@app.cell
def _(pd):
    train_df = pd.read_csv('./data/raw/train.csv')
    test_df = pd.read_csv('./data/raw/test_template.csv')
    return test_df, train_df


@app.cell
def _(train_df):
    # 欠損値チェック
    train_df.isnull().sum()
    return


@app.cell
def _(test_df, train_df):
    # size
    train_df.shape, test_df.shape
    return


@app.cell
def _(train_df):
    train_df.head()
    return


@app.cell
def _(train_df):
    train_df.info()
    return


@app.cell
def _(train_df):
    # 基本情報
    train_df.describe().drop(['id', 'win'], axis=1)
    return


@app.cell
def _(train_df):
    # kills=39のレコードを確認
    train_df[train_df['kills'] == 39]
    return


@app.function
# KDAを計算する関数, 
def calculate_kda(row):
    kills = row['kills']
    deaths = row['deaths']
    assists = row['assists']
    # 分子と分母が0になるのを防ぐ
    kda = (kills + assists + 1) / (deaths + 1)  # avoid division by zero
    return kda


@app.cell
def _(test_df, train_df):
    # KDAのカラムを追加
    train_df['KDA'] = train_df.apply(calculate_kda, axis=1)
    test_df['KDA'] = test_df.apply(calculate_kda, axis=1)
    return


@app.cell
def _(train_df):
    train_df.head()
    return


@app.cell
def _(mo):
    mo.md(r"### 重要そうな要素単体での勝率への寄与率を調べてみる")
    return


@app.cell
def _(train_df):
    train_df[['KDA', 'win']].groupby('win').describe().T
    return


@app.cell
def _(train_df):
    # pentakills>0のレコードについて、winの割合を確認
    train_df[train_df['pentakills'] > 0]['win'].value_counts(normalize=True)
    return


@app.cell
def _(train_df):
    # firstBloodKillの値ごとにwinの割合を確認
    train_df[train_df['firstblood'] == 1]['win'].value_counts(normalize=True)
    return


@app.cell
def _(np, pd, plt, train_df):
    # KDAについて調べる. kda={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10以上}のレコードについて、winの割合を確認
    train_df['KDA_binned'] = pd.cut(train_df['KDA'], 
                                    bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf], 
                                    labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, '10+'])
    train_df[['KDA_binned', 'win']].groupby('KDA_binned').mean().plot(kind='bar')
    plt.title('Win Rate by KDA Binned')
    plt.xlabel('KDA Binned')
    plt.ylabel('Win Rate')
    plt.show()
    return


@app.cell
def _(train_df):
    # kdabinedを削除
    train_df.drop('KDA_binned', axis=1, inplace=True)
    return


@app.cell
def _(train_df):
    # 勝敗の件数
    win_counts = train_df['win'].value_counts()
    print(f"\n勝利: {win_counts[1]} ({win_counts[1]/len(train_df)*100:.2f}%)")
    print(f"敗北: {win_counts[0]} ({win_counts[0]/len(train_df)*100:.2f}%)")
    return (win_counts,)


@app.cell
def _(win_counts):
    win_counts
    return


@app.cell
def _(win_counts):
    # クラスバランスの判定
    balance_ratio = min(win_counts) / max(win_counts)
    print(f"クラスバランス比: {balance_ratio:.2f}")
    return


@app.cell
def _(mo):
    mo.md(r"### 各特徴量の分布を確認")
    return


@app.cell
def _(plt, train_df):
    # id, win列をのぞくカラム
    feature_columns = [col for col in train_df.columns if col not in ['id', 'win']]

    # 数値特徴量の分布をヒストグラムで確認
    n_cols = 5
    n_rows = (len(feature_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for idx, col in enumerate(feature_columns):
        axes[idx].hist(train_df[col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(col)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')

    # 余ったサブプロットを削除
    for idx in range(len(feature_columns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
    return feature_columns, n_cols, n_rows


@app.cell
def _(train_df):
    # 情報量がないカラムについて調べる
    nunique = train_df.nunique()
    constant_columns = nunique[nunique == 1].index.tolist()
    print(f"情報量がないカラム: {constant_columns}")
    return (constant_columns,)


@app.cell
def _(constant_columns, train_df):
    train_df.drop(columns=constant_columns, inplace=True)
    return


@app.cell
def _(feature_columns, n_cols, n_rows, plt, train_df):
    # 箱ひげ図で外れ値を確認
    # feature_columnsからtimeccを削除
    feature_columns_1 = [col for col in feature_columns if col != 'timecc']
    fig_1, axes_1 = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes_1 = axes_1.flatten()
    for idx_1, col_1 in enumerate(feature_columns_1):
        axes_1[idx_1].boxplot(train_df[col_1].dropna(), vert=True)
        axes_1[idx_1].set_title(col_1)
        axes_1[idx_1].set_ylabel('Value')
    for idx_1 in range(len(feature_columns_1), len(axes_1)):
        axes_1[idx_1].axis('off')
    plt.tight_layout()
    # 余分なサブプロットを削除
    plt.show()
    return (feature_columns_1,)


@app.cell
def _(mo):
    mo.md(
        r"""
        memo: 右裾の長いデータを使っているから箱ひげ図は機能しないみたい
        対数変換して再チャレンジ
        """
    )
    return


@app.cell
def _(feature_columns_1, np, plt, test_df, train_df):
    # 全てのカラムを対数変換してみる
    # 新しいデータフレームを作成
    train_df_log = train_df.copy()
    test_df_log = test_df.copy()
    for col_2 in feature_columns_1:
        train_df_log[col_2] = np.log1p(train_df_log[col_2])  # 元の値が0以下の場合、対数変換できないので1を足す
        test_df_log[col_2] = np.log1p(test_df_log[col_2])
    n_cols_1 = 5
    n_rows_1 = (len(feature_columns_1) + n_cols_1 - 1) // n_cols_1
    # 数値特徴量の分布をヒストグラムで確認
    fig_2, axes_2 = plt.subplots(n_rows_1, n_cols_1, figsize=(20, n_rows_1 * 3))
    axes_2 = axes_2.flatten()
    for idx_2, col_2 in enumerate(feature_columns_1):
        axes_2[idx_2].hist(train_df_log[col_2], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes_2[idx_2].set_title(col_2)
        axes_2[idx_2].set_xlabel('Value')
        axes_2[idx_2].set_ylabel('Frequency')
    for idx_2 in range(len(feature_columns_1), len(axes_2)):
        axes_2[idx_2].axis('off')
    # 余ったサブプロットを削除
    plt.tight_layout()
    plt.show()
    return test_df_log, train_df_log


@app.cell
def _(feature_columns_1, plt, test_df, test_df_log, train_df, train_df_log):
    # box-cox変換をしてみる
    from scipy import stats
    # 新しいデータフレームを作成
    train_df_bc = train_df.copy()
    test_df_bc = test_df.copy()
    # box-coxしないカラムのリスト
    keep_columns = ['doublekills', 'triplekills', 'quadrakills', 'pentakills', 'firlstblood']
    feature_columns_2 = [col for col in feature_columns_1 if col not in keep_columns]
    for col_3 in feature_columns_2:
        train_df_bc[col_3] = stats.boxcox(train_df_log[col_3] + 1)[0]
        test_df_bc[col_3] = stats.boxcox(test_df_log[col_3] + 1)[0]  # 元の値が0以下の場合、対数変換できないので1を足す
    n_cols_2 = 5
    n_rows_2 = (len(feature_columns_2) + n_cols_2 - 1) // n_cols_2
    fig_3, axes_3 = plt.subplots(n_rows_2, n_cols_2, figsize=(20, n_rows_2 * 3))
    # 数値特徴量の分布をヒストグラムで確認
    axes_3 = axes_3.flatten()
    for idx_3, col_3 in enumerate(feature_columns_2):
        axes_3[idx_3].hist(train_df_log[col_3], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes_3[idx_3].set_title(col_3)
        axes_3[idx_3].set_xlabel('Value')
        axes_3[idx_3].set_ylabel('Frequency')
    for idx_3 in range(len(feature_columns_2), len(axes_3)):
        axes_3[idx_3].axis('off')
    plt.tight_layout()
    # 余ったサブプロットを削除
    plt.show()
    return (feature_columns_2,)


@app.cell
def _(mo):
    mo.md(r"### 相関分析")
    return


@app.cell
def _(plt, train_df):
    correlations = train_df.corrwith(train_df['win']).sort_values(ascending=False)

    print("勝敗と最も相関が高い特徴量 TOP 10:")
    print(correlations.head(10))

    print("勝敗と最も相関が低い特徴量 TOP 10:")
    print(correlations.tail(10))

    # 可視化
    plt.figure(figsize=(10, 8))
    correlations.plot(kind='barh', color=correlations.apply(lambda x: 'green' if x > 0 else 'red'))
    plt.title('Correlations with Win', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()
    return (correlations,)


@app.cell
def _(mo):
    mo.md(
        r"""
        memo: 
    
        これめちゃくちゃいい。何が勝敗に強く影響を与えているかが一目でわかる。
        ただ、ペンタキルの相関係数がそこまで高くない。ペンタ取ったら9割勝利するので直感に反する。
        どうやら発生頻度が低すぎると相関係数が小さくなるらしい。
        あと自作のKDAの相関が高いのは嬉しい。
    
        相関係数の数式：
        $$
    
        \begin{align*}
        r &= \frac{Cov(X, Y)}{\sigma_X, \sigma_Y} \\
          &= \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}} \\
        \end{align*}
        $$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"pentakillsは分散が小さいから相関係数は大きくなりそうだけど、それ以上に分子の教分散が小さくなるのかな、、？")
    return


@app.cell
def _(mo):
    mo.md(r"なんか分散が小さいと共分散も小さくなるっぽい")
    return


@app.cell
def _(mo):
    mo.md(r"### 勝敗別の特徴量分布比較")
    return


@app.cell
def _(correlations):
    # 重要な特徴量TOP5について、勝敗別の分布を比較
    top_features = correlations.abs().sort_values(ascending=False).head(6).index.tolist()
    top_features.remove('win')  # 'win'を除外
    top_features
    return (top_features,)


@app.cell
def _(plt, top_features, train_df):
    # 重要な特徴量TOP5について、勝敗別の分布を比較（ヒストグラム）
    n_features = len(top_features)
    n_cols_3 = 3
    n_rows_3 = (n_features + n_cols_3 - 1) // n_cols_3
    fig_4, axes_4 = plt.subplots(nrows=n_rows_3, ncols=n_cols_3, figsize=(15, 5 * n_rows_3))
    axes_4 = axes_4.flatten()
    for idx_4, col_4 in enumerate(top_features):  # 1次元配列に変換
        win_data = train_df[train_df['win'] == 1][col_4]
        loss_data = train_df[train_df['win'] == 0][col_4]
        axes_4[idx_4].hist(loss_data, bins=50, alpha=0.5, label='Loss', color='red')  # 勝利と敗北でデータを分ける
        axes_4[idx_4].hist(win_data, bins=50, alpha=0.5, label='Win', color='green')
        axes_4[idx_4].set_title(f'{col_4}', fontsize=12, fontweight='bold')
        axes_4[idx_4].set_xlabel('Value')
        axes_4[idx_4].set_ylabel('Frequency')
        axes_4[idx_4].legend()
        axes_4[idx_4].axvline(loss_data.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
        axes_4[idx_4].axvline(win_data.mean(), color='green', linestyle='--', linewidth=2, alpha=0.7)
    for idx_4 in range(len(top_features), len(axes_4)):
        axes_4[idx_4].axis('off')
    plt.tight_layout()
    # 余分なサブプロットを非表示
    plt.show()  # 平均値の表示
    return


@app.cell
def _(top_features, train_df):
    # 統計的な比較
    for col_5 in top_features:
        win_mean = train_df[train_df['win'] == 1][col_5].mean()
        loss_mean = train_df[train_df['win'] == 0][col_5].mean()
        diff = win_mean - loss_mean
        print(f'{col_5}:')
        print(f'  勝利時平均: {win_mean:.2f}')
        print(f'  敗北時平均: {loss_mean:.2f}')
        print(f'  差: {diff:.2f}\n')
    return


@app.cell
def _(mo):
    mo.md(r"### 特徴量の相関")
    return


@app.cell
def _(train_df):
    type(train_df.iloc[1,1])
    return


@app.cell
def _(feature_columns_2, np, pd, plt, train_df):
    # snsをインポート
    import seaborn as sns
    correlation_matrix = train_df[feature_columns_2].corr()
    # 相関行列を計算
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
    # 高い相関を持つペアを見つける
        for j in range(i + 1, len(correlation_matrix.columns)):
            if np.abs(correlation_matrix.values[i, j]) > 0.8:
                col_i = correlation_matrix.columns[i]
                col_j = correlation_matrix.columns[j]
                high_corr_pairs.append({'Feature 1': correlation_matrix.columns[i], 'Feature 2': correlation_matrix.columns[j], 'Correlation': correlation_matrix.iloc[i, j]})
    if high_corr_pairs:
        print('高い相関を持つ特徴両ペア（｜相関｜> 0.8:')
        high_corr_df = pd.DataFrame(high_corr_pairs)
        print(high_corr_df)
    else:
        print('特に高い相関を持つ特徴両ペアはありません')
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    # ヒートマップで可視化
    plt.show()
    return (correlation_matrix,)


@app.cell
def _(mo):
    mo.md(r"# 相関行列のメモ")
    return


@app.cell
def _(correlation_matrix):
    correlation_matrix.head()
    return


if __name__ == "__main__":
    app.run()
