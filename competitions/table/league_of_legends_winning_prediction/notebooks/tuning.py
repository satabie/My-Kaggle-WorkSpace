import marimo

__generated_with = "0.17.0"
app = marimo.App()


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
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances

    warnings.filterwarnings("ignore")
    return (
        StratifiedKFold,
        accuracy_score,
        lgb,
        np,
        optuna,
        pd,
        plot_optimization_history,
        plot_param_importances,
        roc_auc_score,
        yaml,
    )


@app.cell
def _(pd):
    # データ準備
    X_train = pd.read_csv("./data/processed/train_x_add_features.csv")
    test_df = pd.read_csv("./data/processed/test_x_add_features.csv")

    y_train = X_train["win"]
    X_train.drop(["win"], axis=1, inplace=True)
    X_test = test_df.drop(["id", "win"], axis=1)

    print(X_train.shape, X_test.shape, y_train.shape, test_df.shape)
    return X_test, X_train, test_df, y_train


@app.cell
def _(StratifiedKFold, X_train, yaml):
    config_path = "./configs/config_lgbm_cvmean.yaml"
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    # 交差検証の設定
    n_splits = 5
    optimal_threshold = config["threshold"]["optimal"]
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)

    len(X_train) // n_splits
    return (optimal_threshold,)


@app.cell
def _(StratifiedKFold, X_train, lgb, np, roc_auc_score, y_train):
    # Optunaのobjective関数を定義
    def objective(trial):
        # ハイパーパラメータの探索範囲を定義
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": 42,
            # チューニング対象のパラメータ
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }

        # 交差検証でスコアを計算
        cv_scores = []
        kf_optuna = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)

        for train_idx, valid_idx in kf_optuna.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_idx]
            X_valid_fold = X_train.iloc[valid_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_valid_fold = y_train.iloc[valid_idx]

            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            valid_data = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            y_valid_pred = model.predict(X_valid_fold, num_iteration=model.best_iteration)
            auc = roc_auc_score(y_valid_fold, y_valid_pred)
            cv_scores.append(auc)

        return np.mean(cv_scores)
    return (objective,)


@app.cell
def _(objective, optuna):
    # Optunaでハイパーパラメータ最適化を実行
    study = optuna.create_study(direction="maximize", study_name="lgbm_optimization")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print(f"\nBest trial:")
    print(f"  Value (AUC): {study.best_trial.value:.4f}")
    print(f"  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    return (study,)


@app.cell
def _(plot_optimization_history, plot_param_importances, study):
    # 最適化の履歴を可視化
    fig1 = plot_optimization_history(study)
    fig1.show()

    # パラメータの重要度を可視化
    fig2 = plot_param_importances(study)
    fig2.show()
    return


@app.cell
def _(
    StratifiedKFold,
    X_test,
    X_train,
    accuracy_score,
    lgb,
    np,
    optimal_threshold,
    roc_auc_score,
    study,
    y_train,
):
    # 最適なパラメータで再学習
    best_params = study.best_trial.params
    best_params.update(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": 42,
        }
    )

    # 交差検証で最終評価
    n_splits_final = 5
    kf_final = StratifiedKFold(n_splits=n_splits_final, shuffle=True, random_state=71)

    cv_valid_auc_best = []
    cv_valid_acc_best = []
    oof_predictions_best = np.zeros(X_train.shape[0])
    test_predictions_best = np.zeros(X_test.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(kf_final.split(X_train, y_train), 1):
        print(f"Final Training - Fold {fold} / {n_splits_final}")

        X_train_fold = X_train.iloc[train_idx]
        X_valid_fold = X_train.iloc[valid_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        valid_data = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=train_data)

        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_valid_pred = model.predict(X_valid_fold, num_iteration=model.best_iteration)
        oof_predictions_best[valid_idx] = y_valid_pred  # type: ignore

        test_predictions_best += model.predict(X_test, num_iteration=model.best_iteration) / n_splits_final  # type: ignore

        valid_auc = roc_auc_score(y_valid_fold, y_valid_pred)  # type: ignore
        valid_acc = accuracy_score(y_valid_fold, (y_valid_pred > optimal_threshold).astype(int))  # type: ignore

        cv_valid_auc_best.append(valid_auc)
        cv_valid_acc_best.append(valid_acc)

        print(f"Valid Accuracy: {valid_acc:.4f} | AUC: {valid_auc:.4f}")
        print(f"Best iteration: {model.best_iteration}\n")

    # OOF全体の評価
    oof_auc_best = roc_auc_score(y_train, oof_predictions_best)  # type: ignore
    oof_acc_best = accuracy_score(y_train, (oof_predictions_best > optimal_threshold).astype(int))  # type: ignore

    print(f"\n{'='*50}")
    print(f"Final Results with Best Parameters:")
    print(f"{'='*50}")
    print(f"Valid AUC:      {np.mean(cv_valid_auc_best):.4f} ± {np.std(cv_valid_auc_best):.4f}")
    print(f"Valid Accuracy: {np.mean(cv_valid_acc_best):.4f} ± {np.std(cv_valid_acc_best):.4f}")
    print(f"OOF AUC:        {oof_auc_best:.4f}")
    print(f"OOF Accuracy:   {oof_acc_best:.4f}")
    return (test_predictions_best,)


@app.cell
def _(optimal_threshold, pd, test_df, test_predictions_best):
    # 最適化されたモデルでの提出ファイル作成
    y_test_pred_best = (test_predictions_best > optimal_threshold).astype(int)

    save_path_optuna = "./outputs/submissions/submission_lgbm_optuna.csv"
    submission_optuna = pd.DataFrame({"id": test_df["id"], "win": y_test_pred_best})
    submission_optuna.to_csv(save_path_optuna, index=False)
    print(f"Submission file saved to: {save_path_optuna}")
    return


@app.cell
def _(study, yaml):
    # ベストパラメータを保存
    best_params_to_save = {
        "model": {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": 42,
            **study.best_trial.params,
        },
        "optimization": {
            "best_auc": float(study.best_trial.value),
            "n_trials": len(study.trials),
        },
    }

    save_path = "./configs/config_lgbm_optimized.yaml"
    with open(save_path, "w") as config_out:
        yaml.dump(best_params_to_save, config_out, default_flow_style=False, sort_keys=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
