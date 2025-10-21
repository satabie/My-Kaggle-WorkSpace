import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
    return np, pd, plt, sns, train_test_split


@app.cell
def _(pd, train_test_split):
    # Load the preprocessed data
    train_df = pd.read_csv('../data/processed/train_x_add_features.csv')
    test_df = pd.read_csv('../data/processed/test_x_add_features.csv')

    # 分割
    X = train_df.drop(columns=['win'], axis=1)
    y = train_df['win']
    X_test = test_df.drop(columns=['win'], axis=1)

    # train/validation split
    # stratifyオプション→目的変数の分布を保ったまま分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape
    return X_test, X_train, X_val, test_df, train_df, y_train, y_val


@app.cell
def _(y_train, y_val):
    # 分布の確認
    print("Train win distribution:")
    print(y_train.value_counts(normalize=True))
    print("\nValidation win distribution:")
    print(y_val.value_counts(normalize=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 追加した特徴量の確認""")
    return


@app.cell
def _(plt, train_df):
    # ヒストグラムのプロット
    target_cols = ['kda', 'pure_kill_ratio', 'dmg_per_death', 'dmg_efficiency', 'physical_ratio', 'magic_ratio', 'survival_rate', 'tankiness', 'healing_efficiency', 'tower_dmg_normalized', 'vision_score', 'multiltikill_score', 'overall_performance']
    n_cols = 5
    n_rows = (len(target_cols) + n_cols - 1) // n_cols
    _fig, _axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    _axes = _axes.flatten()
    for idx, col in enumerate(target_cols):
        _axes[idx].hist(train_df[col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        _axes[idx].set_title(col)
        _axes[idx].set_xlabel('Value')
        _axes[idx].set_ylabel('Frequency')
    for idx in range(len(target_cols), len(_axes)):
        _axes[idx].axis('off')
    plt.tight_layout()
    # 余ったサブプロットを削除
    plt.show()
    return


@app.cell
def _(train_df):
    train_df[['dmg_efficiency', 'healing_efficiency']].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### LightGBMモデルの構築（ベースライン）""")
    return


@app.cell
def _(X_train, X_val, plt, y_train, y_val):
    import lightgbm as lgb

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['training', 'valid_0'],
        callbacks=[
            lgb.record_evaluation(evals_result), 
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    # 学習曲線の可視化
    train_loss = evals_result['training']['binary_logloss'][-1]
    valid_loss = evals_result['valid_0']['binary_logloss'][-1]

    print(f'Last train loss: {train_loss:.4f}')
    print(f'Last valid loss: {valid_loss:.4f}')

    # プロット
    lgb.plot_metric(evals_result, metric='binary_logloss', figsize=(10, 6))
    plt.title('Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.legend(['Training', 'Validation'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Best iteration: {model.best_iteration}")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 評価""")
    return


@app.cell
def _():
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
    return (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )


@app.cell
def _(X_train, X_val, model):
    # 予測
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_valid_pred = model.predict(X_val, num_iteration=model.best_iteration)


    optimal_threshold = 0.5078 # 最適閾値
    y_train_pred_binary = (y_train_pred > optimal_threshold).astype(int) # type: ignore
    y_valid_pred_binary = (y_valid_pred > optimal_threshold).astype(int) # type: ignore
    return y_train_pred, y_train_pred_binary, y_valid_pred, y_valid_pred_binary


@app.cell
def _(
    accuracy_score,
    roc_auc_score,
    y_train,
    y_train_pred,
    y_train_pred_binary,
    y_val,
    y_valid_pred,
    y_valid_pred_binary,
):
    # 精度計算
    train_acc = accuracy_score(y_train, y_train_pred_binary)
    valid_acc = accuracy_score(y_val, y_valid_pred_binary)
    train_auc = roc_auc_score(y_train, y_train_pred) # type: ignore
    valid_auc = roc_auc_score(y_val, y_valid_pred) # type: ignore
    return train_acc, train_auc, valid_acc, valid_auc


@app.cell
def _(train_acc, train_auc, valid_acc, valid_auc):
    print(f"\n📊 精度:")
    print(f"  トレーニング Accuracy: {train_acc:.4f}")
    print(f"  検証 Accuracy: {valid_acc:.4f}")
    print(f"  トレーニング AUC: {train_auc:.4f}")
    print(f"  検証 AUC: {valid_auc:.4f}")
    return


@app.cell
def _(classification_report, y_val, y_valid_pred_binary):
    # 詳細なレポート
    print(classification_report(y_val, y_valid_pred_binary, target_names=['Loss', 'Win']))
    return


@app.cell
def _(
    confusion_matrix,
    np,
    plt,
    sns,
    train_acc,
    train_auc,
    valid_acc,
    valid_auc,
    y_val,
    y_valid_pred_binary,
):
    # 混同行列
    cm = confusion_matrix(y_val, y_valid_pred_binary)
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=_axes[0], xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
    _axes[0].set_title('Confusion Matrix')
    _axes[0].set_xlabel('Predicted Label')
    _axes[0].set_ylabel('True Label')
    metrics = ['Accuracy', 'AUC']
    train_scores = [train_acc, train_auc]
    valid_scores = [valid_acc, valid_auc]
    x = np.arange(len(metrics))
    # 精度比較
    width = 0.35
    _axes[1].bar(x - width / 2, train_scores, width, label='Train', color='skyblue')
    _axes[1].bar(x + width / 2, valid_scores, width, label='Valid', color='lightcoral')
    _axes[1].set_ylabel('Score')
    _axes[1].set_title('Model Performance', fontsize=14, fontweight='bold')
    _axes[1].set_xticks(x)
    _axes[1].set_xticklabels(metrics)
    _axes[1].legend()
    _axes[1].set_ylim([0.7, 1.0])
    for i, (train_score, valid_score) in enumerate(zip(train_scores, valid_scores)):
        _axes[1].text(i - width / 2, train_score + 0.01, f'{train_score:.3f}', ha='center', va='bottom', fontsize=10)
        _axes[1].text(i + width / 2, valid_score + 0.01, f'{valid_score:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    # 値をバーの上に表示
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    AUCに対して,ACCが低いので,閾値を最適化する.
    ### Youden's Index
    TPR: True Positive Rate(真陽性率)

    FPR: False Positive Rate(偽陽性率)
    $$
    \begin{align*}
    TPR &= \frac{TP}{TP + FN} \\
    FPR &= \frac{FP}{FP + TN} \\

    optimal\_idx &= \arg\max (TPR - FPR)
    \end{align*}
    $$




    """
    )
    return


@app.cell
def _(accuracy_score, np, y_val, y_valid_pred, y_valid_pred_binary):
    # 閾値の最適化
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_val, y_valid_pred)
    # ROC曲線から最適閾値を探す
    optimal_idx = np.argmax(tpr - fpr)  # type: ignore
    # Youden's Index（TPR - FPR が最大になる点）
    optimal_threshold_1 = thresholds[optimal_idx]
    print(f'最適閾値: {optimal_threshold_1:.4f}')
    y_pred_optimal = (y_valid_pred >= optimal_threshold_1).astype(int)
    optimal_accuracy = accuracy_score(y_val, y_pred_optimal)
    print(f'閾値0.5でのAccuracy: {accuracy_score(y_val, y_valid_pred_binary):.4f}')
    # 最適閾値で予測
    print(f'最適閾値でのAccuracy: {optimal_accuracy:.4f}')
    return (optimal_threshold_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 特徴量重要度の確認""")
    return


@app.cell
def _(train_df):
    feature_columns = train_df.drop(['win'], axis=1).columns
    type(feature_columns)
    return (feature_columns,)


@app.cell
def _(feature_columns, model, pd):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(20).to_string(index=False))
    return (feature_importance,)


@app.cell
def _(feature_importance, plt):
    # 可視化
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature']) # type: ignore
    plt.xlabel('Importance(Gain)', fontsize=12)
    plt.title('Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_test):
    X_test.head()
    return


@app.cell
def _(X_test, model, optimal_threshold_1):
    X_test.drop(['timecc', 'id'], axis=1, inplace=True)
    # テストデータで予測
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_test_pred_binary = (y_test_pred > optimal_threshold_1).astype(int)
    return (y_test_pred_binary,)


@app.cell
def _(y_test_pred_binary):
    print(f"✅ 予測完了")
    print(f"予測されたクラス分布:")
    print(f"  勝利予測: {sum(y_test_pred_binary==1)} ({sum(y_test_pred_binary==1)/len(y_test_pred_binary)*100:.2f}%)")
    print(f"  敗北予測: {sum(y_test_pred_binary==0)} ({sum(y_test_pred_binary==0)/len(y_test_pred_binary)*100:.2f}%)")
    return


@app.cell
def _(model, pd, test_df, y_test_pred_binary):
    import os
    # 提出用ファイル作成
    save_path_submission = '../outputs/submissions'
    save_path_model = '../outputs/models/basemodel_lgbm'
    submission = pd.DataFrame({
        'id': test_df['id'],
        'win': y_test_pred_binary
    })

    os.makedirs(save_path_submission, exist_ok=True)
    os.makedirs(save_path_model, exist_ok=True)
    submission.to_csv(os.path.join(save_path_submission, 'submission_basemodel.csv'), index=False)

    # モデル保存
    model.save_model(os.path.join(save_path_model, 'lgbm_basemodel.txt'))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
