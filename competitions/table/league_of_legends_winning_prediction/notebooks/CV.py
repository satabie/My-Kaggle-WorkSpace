import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    import matplotlib.pyplot as plt
    import yaml
    import warnings

    warnings.filterwarnings("ignore")
    return (
        StratifiedKFold,
        accuracy_score,
        lgb,
        np,
        pd,
        plt,
        roc_auc_score,
        yaml,
    )


@app.cell
def _(pd):
    # データ準備
    X_train = pd.read_csv("./data/processed/train_x_add_features_2.csv")
    test_df = pd.read_csv("./data/processed/test_x_add_features_2.csv")

    y_train = X_train["win"]
    X_train.drop(["win"], axis=1, inplace=True)
    X_test = test_df.drop(["id", "win"], axis=1)

    print(X_train.shape, X_test.shape, y_train.shape, test_df.shape)
    return X_test, X_train, test_df, y_train


@app.cell
def _(StratifiedKFold, yaml):
    config_path = "./configs/config_lgbm_optimized.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 交差検証の設定
    n_splits = 5
    optimal_threshold = config["threshold"]["optimal"]
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)
    return config, config_path, kf, n_splits, optimal_threshold


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    config,
    kf,
    lgb,
    n_splits,
    np,
    optimal_threshold,
    roc_auc_score,
    y_train,
):
    # LightGBMのパラメータ設定
    params = config["model"]["lgbm"]

    cv_train_scores = []
    cv_valid_scores = []
    cv_train_auc = []
    cv_valid_auc = []
    oof_predictions = np.zeros(X_train.shape[0])  # Out-of-Fold predictions
    test_predictions = np.zeros(X_test.shape[0])
    feature_importance_list = []
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train), 1):
        print(f"Fold {fold} / {n_splits}")

        # データ分割
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # LightGBMデータセット作成
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        valid_data = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=train_data)

        # モデル訓練
        model = lgb.train(
            params,
            train_data,
            num_boost_round=config["model"]["learning_control"]["num_boost_round"],
            valid_sets=[train_data, valid_data],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=config["model"]["learning_control"]["early_stopping_rounds"], verbose=False
                ),
                lgb.log_evaluation(period=0),  # ログ出力を無効化
            ],
        )

        # 予測
        y_train_pred = model.predict(X_train_fold, num_iteration=model.best_iteration)
        y_valid_pred = model.predict(X_valid_fold, num_iteration=model.best_iteration)

        # Out-of-Fold予測更新
        oof_predictions[valid_idx] = y_valid_pred  # type: ignore

        # テストデータの予測を累積
        test_predictions += model.predict(X_test, num_iteration=model.best_iteration) / n_splits  # type: ignore

        # 評価
        train_acc = accuracy_score(y_train_fold, (y_train_pred > optimal_threshold).astype(int))  # type: ignore
        valid_acc = accuracy_score(y_valid_fold, (y_valid_pred > optimal_threshold).astype(int))  # type: ignore
        train_auc = roc_auc_score(y_train_fold, y_train_pred)  # type: ignore
        valid_auc = roc_auc_score(y_valid_fold, y_valid_pred)  # type: ignore

        cv_train_scores.append(train_acc)
        cv_valid_scores.append(valid_acc)
        cv_train_auc.append(train_auc)
        cv_valid_auc.append(valid_auc)

        best_iterations.append(model.best_iteration)

        print(f"Train Accuracy: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"Valid Accuracy: {valid_acc:.4f} | AUC: {valid_auc:.4f}")
        print(f"Best iteration: {model.best_iteration}\n")

        # 特徴量重要度の保存
        feature_importance_list.append(model.feature_importance(importance_type="gain"))

    # oof全体の評価
    oof_acc = accuracy_score(y_train, (oof_predictions > optimal_threshold).astype(int))  # type: ignore
    oof_auc = roc_auc_score(y_train, oof_predictions)  # type: ignore
    print(f"OOF Accuracy: {oof_acc:.4f} | AUC: {oof_auc:.4f}")
    return (
        best_iterations,
        cv_train_auc,
        cv_train_scores,
        cv_valid_auc,
        cv_valid_scores,
        oof_acc,
        oof_auc,
        test_predictions,
    )


@app.cell
def _(
    best_iterations,
    config_path,
    cv_valid_auc,
    cv_valid_scores,
    oof_acc,
    oof_auc,
):
    import sys

    sys.path.append("./src")
    from ExperimentLogger import ExperimentLogger

    exp_logger = ExperimentLogger(config_path=config_path)
    exp_logger.log_experiment(
        cv_valid_auc=cv_valid_auc,
        cv_valid_acc=cv_valid_scores,
        oof_auc=oof_auc,
        oof_acc=oof_acc,
        best_iterations=best_iterations,  # オプション
    )
    return


@app.cell
def _(
    cv_train_auc,
    cv_train_scores,
    cv_valid_auc,
    cv_valid_scores,
    n_splits,
    np,
    oof_acc,
    oof_auc,
    pd,
):
    # 各Foldの結果
    results_df = pd.DataFrame(
        {
            "Fold": range(1, n_splits + 1),
            "Train Acc": cv_train_scores,
            "Valid Acc": cv_valid_scores,
            "Train AUC": cv_train_auc,
            "Valid AUC": cv_valid_auc,
        }
    )

    print(results_df.to_string(index=False))

    # 平均と標準偏差
    print(f"\n{'='*50}")
    print(f"平均と標準偏差:")
    print(f"{'='*50}")
    print(f"Valid Accuracy: {np.mean(cv_valid_scores):.4f} ± {np.std(cv_valid_scores):.4f}")
    print(f"Valid AUC:      {np.mean(cv_valid_auc):.4f} ± {np.std(cv_valid_auc):.4f}")

    print(f"\n{'='*50}")
    print(f"Out-of-Fold 全体評価:")
    print(f"{'='*50}")
    print(f"OOF Accuracy: {oof_acc:.4f}")
    print(f"OOF AUC:      {oof_auc:.4f}")
    return


@app.cell
def _(
    cv_train_auc,
    cv_train_scores,
    cv_valid_auc,
    cv_valid_scores,
    n_splits,
    np,
    plt,
):
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 各Foldのスコア
    folds = range(1, n_splits + 1)
    axes[0].plot(folds, cv_train_scores, "o-", label="Train Acc", linewidth=2, markersize=8)
    axes[0].plot(folds, cv_valid_scores, "s-", label="Valid Acc", linewidth=2, markersize=8)
    axes[0].axhline(
        np.mean(cv_valid_scores),
        color="red",
        linestyle="--",
        label=f"Mean Valid: {np.mean(cv_valid_scores):.4f}",
        alpha=0.7,
    )
    axes[0].set_xlabel("Fold", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Accuracy across Folds", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(folds)

    # AUC
    axes[1].plot(folds, cv_train_auc, "o-", label="Train AUC", linewidth=2, markersize=8)
    axes[1].plot(folds, cv_valid_auc, "s-", label="Valid AUC", linewidth=2, markersize=8)
    axes[1].axhline(
        np.mean(cv_valid_auc), color="red", linestyle="--", label=f"Mean Valid: {np.mean(cv_valid_auc):.4f}", alpha=0.7
    )
    axes[1].set_xlabel("Fold", fontsize=12)
    axes[1].set_ylabel("AUC", fontsize=12)
    axes[1].set_title("AUC across Folds", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(folds)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(optimal_threshold, pd, test_df, test_predictions):
    # テストデータの最終予測(全Fold平均)
    y_test_pred_binary = (test_predictions > optimal_threshold).astype(int)

    save_path = "./outputs/submissions/submission_lgbm_optuna_add_feature.csv"
    submission_df = pd.DataFrame({"id": test_df["id"], "win": y_test_pred_binary})
    submission_df.to_csv(save_path, index=False)
    return


@app.cell
def _(cv_train_auc, cv_train_scores, cv_valid_auc, cv_valid_scores, np):
    # 過学習の度合いをチェック
    train_valid_gap = np.mean(cv_train_scores) - np.mean(cv_valid_scores)
    train_valid_gap_auc = np.mean(cv_train_auc) - np.mean(cv_valid_auc)
    return train_valid_gap, train_valid_gap_auc


@app.cell
def _(train_valid_gap, train_valid_gap_auc):
    print(train_valid_gap, train_valid_gap_auc)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
