import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from feature_engineering import feature_engineering
from prepare_data_for_models import splitting_data
import torch, joblib, json
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
from pytorch_forecasting.data import GroupNormalizer

plt.switch_backend("Agg")

# Root project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
MODEL_REPORTS = os.path.join(PROJECT_ROOT, 'reports/TFT')
MODEL_FOLDER = os.path.join(PROJECT_ROOT, 'models/TFT')

# Dataloaders
BATCH_SIZE = 256

def load_tft_model():
    # Automatically find and load the latest checkpoint
    # Find all checkpoint files 
    checkpoint_files = glob.glob(os.path.join(MODEL_FOLDER, '*.ckpt'))
    
    if not checkpoint_files:
        raise FileNotFoundError(f'No checkpoints files found in {MODEL_FOLDER}')
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getatime, reverse=True)

    # Get the latest checkpoint
    latest_chekpoint = checkpoint_files[0]
    print(f'Found {len(checkpoint_files)} checkpoint files.')
    print(f'Loading latest: {os.path.basename(latest_chekpoint)}')

    # Load model
    model = TemporalFusionTransformer.load_from_checkpoint(latest_chekpoint)
    model.eval()
    print("✅ TFT model loaded successfully.")

    return model


def load_fine_tuned_tft_model():
    # Automatically find and load the latest checkpoint
    # Find all checkpoint files 
    fine_tuned_folder = os.path.join(MODEL_FOLDER, 'finetuned')
    checkpoint_files = glob.glob(os.path.join(fine_tuned_folder, '*.ckpt'))
    
    if not checkpoint_files:
        raise FileNotFoundError(f'No checkpoints files found in {MODEL_FOLDER}')
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getatime, reverse=True)

    # Get the latest checkpoint
    latest_chekpoint = checkpoint_files[0]
    print(f'Found {len(checkpoint_files)} checkpoint files.')
    print(f'Loading latest: {os.path.basename(latest_chekpoint)}')

    # Load model
    model = TemporalFusionTransformer.load_from_checkpoint(latest_chekpoint)
    model.eval()
    print("✅ TFT-Tuned model loaded successfully.")

    return model

def save_results(results, fine_tuned=False):
    
    # Convert ALL values to ensure they're serializable
    results_converted = {}
    for key, value in results.items():
        if hasattr(value, 'item'):  # numpy types
            results_converted[key] = value.item()
    
    if not fine_tuned:
        with open(os.path.join(MODEL_FOLDER, 'TFT_metrics.json'), 'w') as f:
            json.dump(results_converted, f, indent=4)
    else:
        fine_tuned_folder = os.path.join(MODEL_FOLDER, 'finetuned')
        os.makedirs(fine_tuned_folder, exist_ok=True)
        with open(os.path.join(fine_tuned_folder, 'TFT_tuned_metrics.json'), 'w') as f:
            json.dump(results_converted, f, indent=4)

    print("✅ Result saved successfully.")


def save_training(training: TimeSeriesDataSet):
    joblib.dump(training, os.path.join(MODEL_FOLDER, 'traning.joblib'))
    print("✅ TFT training saved successfully.")

def load_training():
    training = joblib.load(os.path.join(MODEL_FOLDER, 'traning.joblib'))
    return training

def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str, list, list, list, list]:
    df_tft = df

    # Dropping engineered history features
    drop_cols = [c for c in df_tft.columns if c.startswith(('lag_1','diff_'))]
    df_tft = df_tft.drop(columns=drop_cols)

    # Adding required fields 
    df_tft = df_tft.sort_values('datetime').reset_index(drop=True)
    df_tft['time_idx'] = np.arange(len(df_tft), dtype=np.int32)
    df_tft['series_id'] = 'series_1'                                # single series id
     
    # ensure types are numeric
    for c in ["holiday","school","covid_period","is_weekend","hour","dayofweek","month","year","trend"]:
        if c in df_tft.columns:
            df_tft[c] = pd.to_numeric(df_tft[c])

    # Target
    target = 'nat_demand'

    # Future information we already know 
    known_future_reals = [c for c in [
        "hour","dayofweek","is_weekend","month_sin","month_cos","dow_sin","dow_cos",
        "holiday","school","covid_period","trend"
    ] if c in df_tft.columns]

    # Information we know after it happens 
    observed_past_reals = [c for c in [
        target, "weather_pc1","weather_pc2","weather_pc3","TQL_toc","TQL_san","TQL_dav", 'lag_24'
    ] if c in df_tft.columns]

    # Cleaner static feature handling
    static_categoricals = []  
    static_reals = []

    # Validate checks
    validate_cyclical_features(df_tft)
    validate_no_data_leaks(known_future_reals, observed_past_reals, target)

    print(f"✅ Prepared data: {len(df_tft)} rows, {len(known_future_reals)} known future, {len(observed_past_reals)} observed past")

    return df_tft, target, known_future_reals, observed_past_reals, static_categoricals, static_reals

def eval_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_tft = df

    # Dropping engineered history features
    drop_cols = [c for c in df_tft.columns if c.startswith(('lag_1','diff_'))]
    df_tft = df_tft.drop(columns=drop_cols)

    # Adding required fields 
    df_tft = df_tft.sort_values('datetime').reset_index(drop=True)
    df_tft['time_idx'] = np.arange(len(df_tft), dtype=np.int32)
    df_tft['series_id'] = 'series_1'                                # single series id
     
    # ensure types are numeric
    for c in ["holiday","school","covid_period","is_weekend","hour","dayofweek","month","year","trend"]:
        if c in df_tft.columns:
            df_tft[c] = pd.to_numeric(df_tft[c])


    return df_tft

def validate_cyclical_features(df):
    """Ensure cyclical features are properly scaled"""
    cyclical_pairs = [("month_sin", "month_cos"), ("dow_sin", "dow_cos")]
    for sin_feat, cos_feat in cyclical_pairs:
        if sin_feat in df.columns and cos_feat in df.columns:
            # Check they form a unit circle
            magnitude = np.sqrt(df[sin_feat]**2 + df[cos_feat]**2)
            assert np.allclose(magnitude, 1.0, atol=0.1), \
                f"Cyclical features {sin_feat}, {cos_feat} not on unit circle"

def validate_no_data_leaks(known_future, observed_past, target):
    """Ensure no feature misclassification"""
    # Check no overlap between categories
    overlap = set(known_future) & set(observed_past)
    assert len(overlap) == 0, f"Feature overlap between categories: {overlap}"
    
    # Check target is in observed past
    assert target in observed_past, f"Target {target} must be in observed_past_reals"

def mask(df_tft: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:

    train_start = train_df["datetime"].iloc[0]
    train_end   = train_df["datetime"].iloc[-1]
    val_start   = val_df["datetime"].iloc[0]
    val_end     = val_df["datetime"].iloc[-1]
    test_start  = test_df["datetime"].iloc[0]
    test_end    = test_df["datetime"].iloc[-1]

    # Verify no overlaps
    assert train_end < val_start, "Train and validation overlap!"
    assert val_end < test_start, "Validation and test overlap!"

    def mask_between(s, a, b):
        return (s >= a) & (s <= b)

    train_mask = mask_between(df_tft["datetime"], train_start, train_end)
    val_mask   = mask_between(df_tft["datetime"], val_start, val_end)
    test_mask  = mask_between(df_tft["datetime"], test_start, test_end)

    # Verify coverage and no gaps
    total_masked = train_mask.sum() + val_mask.sum() + test_mask.sum()
    assert total_masked == len(df_tft), f"Missing data! Covered {total_masked}/{len(df_tft)} rows"

    return train_mask, val_mask, test_mask

def build_kwargs(target: str, static_categoricals: list, known_future_reals: list, observed_past_reals: list, static_reals: list,
                           max_encoder_length: int=168, max_prediction_length: int=24) -> dict:
    max_encoder_length = max_encoder_length         # 1 week history
    max_prediction_length = max_prediction_length   # predict next 24 hours

    common_kwargs = dict(
        time_idx="time_idx",
        target=target,
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known_future_reals,
        time_varying_unknown_reals=observed_past_reals,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation='softplus'),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    return common_kwargs

def create_dataloaders(df_tft: pd.DataFrame, train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series, common_kwargs: dict):

    # 1. Training dataset (predict=False for robustness)
    training = TimeSeriesDataSet(
        df_tft.loc[train_mask],
        **common_kwargs,
    )

    # 2. Validation dataset (predict=True for honest evaluation)
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df_tft.loc[val_mask],
        predict=True,
        stop_randomization=True
    )

    # 3. Test dataset (predict=True for honest evaluation)
    test = TimeSeriesDataSet.from_dataset(
        training,
        df_tft.loc[test_mask],
        predict=True,
        stop_randomization=True
    )

    num_workers = min(4, os.cpu_count())  # Use available CPU cores, max 8      # type: ignore
    
    print(f"📊 Using {num_workers} workers for data loading")

    train_loader = training.to_dataloader(
        train=True, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = test.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True
    )

    save_training(training)

    return training, validation, test, train_loader, val_loader, test_loader

def create_TFT(training: TimeSeriesDataSet):

    # Model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        # Optimization
        learning_rate=1e-3,
        weight_decay=1e-4, 

        # Architecture
        hidden_size=64,                 # Main network width
        attention_head_size=4,          # Attention heads
        hidden_continuous_size=16,      # Continuous variable encoding

        # Regularization
        dropout=0.2,                    # Prevent overfitting

        # Output configuration
        output_size=7,                  # QuantileLoss intervals
        loss=QuantileLoss(),            # Uncertainty quantification

        # Training monitoring
        log_interval=100,               # Log every 50 batches
        reduce_on_plateau_patience=4    # LR reduction patience
    )
    # tft.save_hyperparameters(ignore=["loss", "logging_metrics"])

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',      # Primary metric for model performance
        patience=6,              # Balance between convergence and overfitting
        mode='min',              # Objective: minimize validation loss
        min_delta=0.005,         # Ignore improvements < 0.001
        verbose=True             # Print when early stopping triggers
    )

    ckpt = ModelCheckpoint(
        dirpath=MODEL_FOLDER,           # Organized model storage
        filename='tft-best-{epoch:02d}-{val_loss:.4f}',  # Descriptive naming
        save_top_k=1,                   # Storage efficiency - keep only best
        monitor='val_loss',             # Consistent with early stopping
        mode='min',                     # Consistent optimization direction
        save_weights_only=False,        # Save entire model (architecture + weights)
        every_n_epochs=1                # Check for improvements every epoch
    )

    lr_logger = LearningRateMonitor(
        logging_interval='epoch'        # Essential for learning rate analysis
    )

    trainer = Trainer(
        # Training Duration
        max_epochs=30,               # Training budget - early stopping may stop before this
       
        # Hardware Configuration
        accelerator='auto',          # Automatic hardware detection (GPU/CPU)
        devices='auto',              # Use all available compute resources

        # Training Stability
        gradient_clip_val=0.5,       # Clip gradients to stabilize transformer training

        # Training Automation
        callbacks=[early_stop, ckpt, lr_logger],  # Training automation and monitoring

        # Monitoring & Logging
        log_every_n_steps=50,        # Frequent logging for training progress visibility

        # Optional Enhancements
        enable_progress_bar=True,    # Visual training progress
        enable_model_summary=True,   # Print model architecture at start
    )

    return tft, trainer, ckpt

def fit_tft(trainer, train_loader, val_loader, ckpt, tft):

    print("🚀 Starting TFT training...")
    trainer.fit(tft, train_loader, val_loader)

    best_path = ckpt.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)
    print("Best checkpoint: ", best_path)

    return best_tft

def evaluate(loader, model):    
    # Evaluate TFT model on a dataloader
    # Args:
    #       loader: DataLoader with test/validation data
    #       model: Trained TFT model

    preds, y = [], []
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Extract the dictionary from the list
            # batch is a list: [data_dict, extra_data]
            batch_dict = batch[0]  # This is the dictionary we need
            batch_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_dict.items()}
            
            out = model(batch_dict)
            
            prediction_tensor = out.prediction
            
            n_quantiles = model.hparams.output_size
            median_idx = n_quantiles // 2

            p50 = prediction_tensor[..., median_idx].detach().cpu().numpy()
            targets = batch_dict['decoder_target'].cpu().numpy()

            preds.append(p50)
            y.append(targets)

            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1} batches...")


    # Concatenate all batches
    preds = np.concatenate(preds, axis=0).reshape(-1)
    y = np.concatenate(y, axis=0).reshape(-1)

    # Calculate metrics
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)

    # MAPE calculation
    mask = y != 0
    if np.any(mask):
        mape = np.mean(np.abs((y[mask] - preds[mask]) / y[mask]) * 100)
    else:
        mape = float('inf')
    
    return mae, rmse, mape, y, preds

def evaluation_TFT(model, X: pd.DataFrame, verbose=True):    
    # Evaluation for future data
    # Args:
    #       loader: DataLoader with test/validation data
    #       model:  TFT or TFT_Tuned model
    df = eval_prepare_dataframe(X)
    training = load_training()
    
    preds, y = [], []
    model.eval()
    device = next(model.parameters()).device

    X_tsd = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    loader = X_tsd.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=4)
    try:    
        with torch.no_grad():
            for i, batch in enumerate(loader):
                # Extract the dictionary from the list
                # batch is a list: [data_dict, extra_data]
                batch_dict = batch[0]  # This is the dictionary we need
                batch_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_dict.items()}
                
                out = model(batch_dict)
                
                prediction_tensor = out.prediction
                
                n_quantiles = model.hparams.output_size
                median_idx = n_quantiles // 2

                print("Prediction tensor shape:", prediction_tensor.shape)
                print("Output size (quantiles):", n_quantiles)
                print("Median index:", median_idx)

                # Assuming shape (batch, horizon, quantiles)
                p50 = prediction_tensor[:, :, median_idx].detach().cpu().numpy()

                preds.append(p50)
                print(f"prediction_tensor shape: {prediction_tensor.shape}, n_quantiles: {n_quantiles}, median_idx: {median_idx}")
                

                # only collect y if it exists
                if 'decoder_target' in batch_dict:
                    targets = batch_dict['decoder_target'].cpu().numpy()
                    y.append(targets)

                # Progress indicator
                if (i + 1) % 20 == 0 & verbose:
                    print(f"   Processed {i + 1} batches...")
    except Exception as e:
        print("Error: ", e)

    # Concatenate all batches
    preds = np.concatenate(preds, axis=0).reshape(-1)
    if len(y) > 0:
        print(f"\n✅ Completed evaluation on {len(preds)} predictions (with ground truth).")
        return preds
    else:
        print(f"\n✅ Completed forecasting {len(preds)} future predictions (no ground truth available).")
        return preds

def move_batch_to_device(batch, device):
    """
    Move all tensors in batch dictionary to specified device
    """
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch

def plot_predictions(test_loader, model, y_test_true, y_test_pred, fine_tuned=False, N=1000):
    """
    Plot actual vs predicted values with uncertainty intervals
    """
    # Last N predictions (flattened sequence-wise)
    plt.figure(figsize=(15, 6))
    
    # Plot actual and predicted values
    plt.plot(y_test_true[-N:], label="Actual", lw=2, color='blue')
    plt.plot(y_test_pred[-N:], label="TFT P50 (Median)", alpha=0.8, color='red')
    
    # Get P10 / P90 to shade uncertainty intervals
    def collect_quantiles(loader, model, q_index):
        quantiles_all = []
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for batch in loader:
                batch_dict = batch[0]  # Extract dictionary from list
                batch_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_dict.items()}
                
                out = model(batch_dict)
                
                # Use the correct prediction attribute based on your debug
                if hasattr(out, 'prediction'):
                    prediction_tensor = out.prediction
                elif hasattr(out, 'output'):
                    prediction_tensor = out.output
                else:
                    prediction_tensor = out
                
                quantile_values = prediction_tensor[..., q_index].cpu().numpy()
                quantiles_all.append(quantile_values)
        
        return np.concatenate(quantiles_all, axis=0).reshape(-1)
    
    # For output_size=7, the quantile indices are:
    # 0: 0.02, 1: 0.1, 2: 0.25, 3: 0.5, 4: 0.75, 5: 0.9, 6: 0.98
    p10 = collect_quantiles(test_loader, model, q_index=1)  # ~0.1 quantile
    p90 = collect_quantiles(test_loader, model, q_index=5)  # ~0.9 quantile
    
    # Plot uncertainty interval
    plt.fill_between(range(len(y_test_pred[-N:])),
                     p10[-N:], p90[-N:], color="orange", alpha=0.3, label="P10–P90 Uncertainty")
    
    plt.title(f"TFT — Test Forecast (last {N} points)", fontsize=14, fontweight='bold')
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Target Value", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_REPORTS, f"TFT — Test Forecast (last {N} points).png"))
    
    # Print some statistics
    print(f"📊 Prediction Interval Statistics (last {N} points):")
    print(f"   - P10 (10th percentile): {p10[-N:].mean():.2f} ± {p10[-N:].std():.2f}")
    print(f"   - P50 (Median): {y_test_pred[-N:].mean():.2f} ± {y_test_pred[-N:].std():.2f}")
    print(f"   - P90 (90th percentile): {p90[-N:].mean():.2f} ± {p90[-N:].std():.2f}")
    print(f"   - Uncertainty range (P90-P10): {(p90[-N:] - p10[-N:]).mean():.2f}")

def evalute_val_test(val_loader, test_loader, model, fine_tuned=False):
    val_mae, val_rmse, val_mape, y_val_true, y_val_pred = evaluate(val_loader, model)
    test_mae, test_rmse, test_mape, y_test_true, y_test_pred = evaluate(test_loader, model)

    if not fine_tuned:
        print(f'Validation -> MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, MAPE: {val_mape:.2f}%')
        print(f'Test       -> MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, MAPE {test_mape:.2f}%')

    else:
        print(f'FIned-tuned Validation -> MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, MAPE: {val_mape:.2f}%')
        print(f'FIned-tuned Test       -> MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, MAPE {test_mape:.2f}%')


    results = {
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_mape": val_mape,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_mape": test_mape
    }

    plot_predictions(test_loader, model, y_test_true, y_test_pred, fine_tuned)

    save_results(results, fine_tuned)

    return results

def fine_tune(df_tft, training, model):
    # 1) After pretraining:
    best_tft = model

    # 2) Build masks for FT and HOLDOUT inside the old test range
    ft_start = pd.Timestamp("2020-01-01 00:00:00")
    ft_end   = pd.Timestamp("2020-05-15 23:00:00")
    ho_start = pd.Timestamp("2020-05-16 00:00:00")
    ho_end   = pd.Timestamp("2020-06-27 00:00:00")

    ft_mask  = (df_tft["datetime"] >= ft_start) & (df_tft["datetime"] <= ft_end)
    ho_mask  = (df_tft["datetime"] >= ho_start) & (df_tft["datetime"] <= ho_end)

    # 3) Reuse your common_kwargs built from training dataset
    #    Create a *fine-tune* dataset from your already-built training dataset
    ft_dataset = TimeSeriesDataSet.from_dataset(training, df_tft.loc[ft_mask], predict=False, stop_randomization=True)

    # For early stopping during fine-tuning, split FT window into a small val
    # e.g., last 10–20% of ft_window as ft_val
    ft_cut = int(0.85 * len(df_tft.loc[ft_mask]))
    ft_train_idx = df_tft.loc[ft_mask].index[:ft_cut]
    ft_val_idx   = df_tft.loc[ft_mask].index[ft_cut:]

    ft_train_ds = TimeSeriesDataSet.from_dataset(training, df_tft.loc[ft_train_idx], predict=False, stop_randomization=True)
    ft_val_ds   = TimeSeriesDataSet.from_dataset(training, df_tft.loc[ft_val_idx],   predict=True,  stop_randomization=True)

    ft_train_loader = ft_train_ds.to_dataloader(train=True,  batch_size=64, num_workers=0, pin_memory=True)
    ft_val_loader   = ft_val_ds.to_dataloader(  train=False, batch_size=64, num_workers=0, pin_memory=True)

    # 4) Make a HOLDOUT dataloader (never used during fine-tune)
    holdout_ds = TimeSeriesDataSet.from_dataset(training, df_tft.loc[ho_mask], predict=True, stop_randomization=True)
    holdout_loader = holdout_ds.to_dataloader(train=False, batch_size=64, num_workers=0, pin_memory=True)

    # 5) Fine-tune with tiny LR and short patience
    best_tft.hparams.learning_rate = 1e-4  # shrink LR

    ft_ckpt = ModelCheckpoint(
        dirpath=os.path.join(MODEL_FOLDER, "finetuned"),
        filename="tft-finetuned-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1, monitor="val_loss", mode="min"
    )
    ft_es = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    ft_trainer = Trainer(
        max_epochs=10, gradient_clip_val=0.5,
        callbacks=[ft_es, ft_ckpt], enable_progress_bar=True
    )

    ft_trainer.fit(best_tft, ft_train_loader, ft_val_loader)

    # Load finetuned best and evaluate ONLY on holdout
    ft_best = TemporalFusionTransformer.load_from_checkpoint(ft_ckpt.best_model_path)

    # Reuse your evaluate() helper
    ho_mae, ho_rmse, ho_mape, y_ho, yhat_ho = evaluate(holdout_loader, ft_best)
    print(f"HOLDOUT (never seen during fine-tune) -> MAE: {ho_mae:.2f}, RMSE: {ho_rmse:.2f}, MAPE: {ho_mape:.2f}%")


def main():
    
    # Loading cleaned data
    df = pd.read_csv(os.path.join(DATA_FOLDER, 'worked/worked_dataset.csv'), parse_dates=['datetime'])
    df = df.sort_values('datetime')

    train_df, val_df, test_df = splitting_data(df)# Splitting data for model

    # Feature engineering with no data leakege
    train_df, scaler, pca = feature_engineering(train_df)
    val_df, _, _ = feature_engineering(val_df, scaler=scaler, pca=pca)
    test_df, _, _ = feature_engineering(test_df, scaler=scaler, pca=pca)

    # Prepare data for model
    train_df, target, known_future_reals, observed_past_reals, static_categoricals, static_reals= prepare_dataframe(train_df)
    val_df, _, _, _, _, _= prepare_dataframe(val_df)
    test_df, _, _, _, _, _= prepare_dataframe(test_df)

    df_tft = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    # masking the data
    train_mask, val_mask, test_mask = mask(df_tft, train_df, val_df, test_df)

    # Creating common kwargs for training
    common_kwargs = build_kwargs(target, static_categoricals, known_future_reals, observed_past_reals, static_reals)

    # Creating complete datasets for training the model
    training, validation, test, train_loader, val_loader, test_loader = create_dataloaders(df_tft, train_mask, val_mask, test_mask, common_kwargs)

    # Creating TFT
    tft, trainer, ckpt = create_TFT(training)

    # Fitting
    model = fit_tft(trainer, train_loader, val_loader, ckpt, tft)

    # Loading model
    # model = load_tft_model()
    results = evalute_val_test(val_loader, test_loader, model)

    # model = load_fine_tuned_tft_model()

    model.train()
    fine_tune(df_tft, training, model)
    
    results_tuned = evalute_val_test(val_loader, test_loader, model, True)



if __name__ == '__main__':
    main()