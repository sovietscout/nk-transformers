import argparse
import gc
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import tomllib
import warnings

from src.simulator import load_and_prepare, build_y_only_dataset
from src.train import train_model
from src.evaluate import (
    collect_irf_paths,
    evaluate_bvar_multistep,
    evaluate_bvar_one_step,
    evaluate_irf_accuracy,
    evaluate_kalman_one_step,
    evaluate_multistep_mse,
    evaluate_one_step_mse,
    evaluate_var_multistep,
    evaluate_var_one_step,
)
from src.visualisation import (
    plot_forecast_horizon,
    plot_irf_grid,
    plot_irf_paths,
    plot_learning_curve,
    plot_reduced_form_trajectory_overlay,
    plot_trajectory_overlay,
)

# Suppress noisy compiler logs and warnings
torch._logging.set_logs(recompiles=False, dynamic=False)
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")


def parse_args():
    p = argparse.ArgumentParser(description='NK Transformer — Replication + Extensions')
    p.add_argument('--config', type=str, default='config.toml',
                  help='Path to TOML config file')
    p.add_argument('--skip-train', action='store_true',
                   help='Skip training, load existing checkpoint')
    p.add_argument('--skip-benchmarks', action='store_true',
                   help='Skip VAR/BVAR benchmarks (slow)')
    p.add_argument('--skip-yonly', action='store_true',
                   help='Skip y-only Transformer vs Kalman experiment')
    return p.parse_args()


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'rb') as f:
        cfg = tomllib.load(f)

    required_keys = {
        'paths': ['cache', 'checkpoints', 'figures'],
        'training': ['epochs', 'batch_size', 'learning_rate', 'compile'],
        'experiment': ['policy_holdout', 'n_irf', 'sample_sizes'],
    }
    for section, keys in required_keys.items():
        if section not in cfg:
            raise KeyError(f"Missing required config section: {section}")
        for key in keys:
            if key not in cfg[section]:
                raise KeyError(f"Missing required config key: {section}.{key}")

    cfg['paths'] = {k: Path(v) for k, v in cfg['paths'].items()}

    return cfg


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, '__fspath__'):
        return str(obj)
    return obj


def main():
    total_t0 = time.perf_counter()
    cli_args = parse_args()
    cfg = load_config(cli_args.config)

    cfg['skip_train'] = cli_args.skip_train
    cfg['skip_benchmarks'] = cli_args.skip_benchmarks
    cfg['skip_yonly'] = cli_args.skip_yonly

    cfg['paths']['cache'].mkdir(parents=True, exist_ok=True)
    cfg['paths']['checkpoints'].mkdir(parents=True, exist_ok=True)
    cfg['paths']['figures'].mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] NK-Transformers | Policy: {cfg['experiment']['policy_holdout']}")
    t1 = time.perf_counter()
    data, stats = load_and_prepare(
        cfg['paths']['cache'],
        policy_holdout=cfg['experiment']['policy_holdout']
    )

    t2 = time.perf_counter()

    device = cfg['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print(f"[WARN] CUDA unavailable; using CPU")

    if not cfg['skip_train']:
        model, history = train_model(
            data=data,
            device=device,
            epochs=cfg['training']['epochs'],
            batch_size=cfg['training']['batch_size'],
            lr=cfg['training']['learning_rate'],
            checkpoint_dir=cfg['paths']['checkpoints'],
            compile_model=cfg['training']['compile'],
        )
        with open(cfg['paths']['checkpoints'] / 'history.pkl', 'wb') as f:
            pickle.dump(history, f)
    else:
        from src.model import NKTransformer
        model = NKTransformer(d_model=64, n_heads=4, n_layers=4, ff_dim=256, dropout=0.1)
        ckpt_path = cfg['paths']['checkpoints'] / 'best_model.pt'
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        model.eval()

    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    t3 = time.perf_counter()

    one_step_results = {}
    multistep_results = {}

    tf_one_mse, tf_one_pervar = evaluate_one_step_mse(model, data, stats, device=device)
    one_step_results['Transformer'] = (tf_one_mse, tf_one_pervar)

    tf_ms_mse = evaluate_multistep_mse(model, data, stats, device=device)
    multistep_results['Transformer'] = tf_ms_mse

    if not cfg['skip_benchmarks']:
        try:
            var_one_mse, var_one_pervar = evaluate_var_one_step(data, stats)
            one_step_results['VAR'] = (var_one_mse, var_one_pervar)
        except Exception as e:
            print(f"[ERROR] VAR one-step: {e}")

        try:
            var_ms_mse = evaluate_var_multistep(data, stats)
            multistep_results['VAR'] = var_ms_mse
        except Exception as e:
            print(f"[ERROR] VAR multi-step: {e}")

        try:
            bvar_one_mse, bvar_one_pervar = evaluate_bvar_one_step(data, stats)
            one_step_results['BVAR'] = (bvar_one_mse, bvar_one_pervar)
        except Exception as e:
            print(f"[ERROR] BVAR one-step: {e}")

        try:
            bvar_ms_mse = evaluate_bvar_multistep(data, stats)
            multistep_results['BVAR'] = bvar_ms_mse
        except Exception as e:
            print(f"[ERROR] BVAR multi-step: {e}")

    yonly_results = {}
    if not cfg['skip_yonly']:
        t3b = time.perf_counter()

        y_data, y_stats = build_y_only_dataset(data, stats)
        y_ckpt = cfg['paths']['checkpoints'] / 'y_only'

        if not cfg['skip_train']:
            y_model, y_history = train_model(
                data=y_data,
                device=device,
                epochs=cfg['training']['epochs'],
                batch_size=cfg['training']['batch_size'],
                lr=cfg['training']['learning_rate'],
                checkpoint_dir=y_ckpt,
                compile_model=cfg['training']['compile'],
            )
            with open(y_ckpt / 'history.pkl', 'wb') as f:
                pickle.dump(y_history, f)
        else:
            from src.model import NKTransformer
            y_model = NKTransformer(d_model=64, n_heads=4, n_layers=4,
                                    ff_dim=256, dropout=0.1,
                                    input_dim=3, output_dim=3)
            y_ckpt_path = y_ckpt / 'best_model.pt'
            y_model.load_state_dict(torch.load(y_ckpt_path, map_location='cpu', weights_only=True))
            y_model.eval()

        y_tf_mse, y_tf_pervar = evaluate_one_step_mse(y_model, y_data, y_stats, device=device)
        kalman_mse, kalman_pervar = evaluate_kalman_one_step(data)
        yonly_results = {
            'Y-only Transformer': (y_tf_mse, y_tf_pervar),
            'Kalman VAR': (kalman_mse, kalman_pervar),
        }

        del y_model
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    t4 = time.perf_counter()
    irf_summary = evaluate_irf_accuracy(model, data, stats,
                                        n_sims=cfg['experiment']['n_irf'], device=device)

    t5 = time.perf_counter()
    sample_sizes = cfg['experiment']['sample_sizes']
    tf_mse_by_n = []
    times_by_n = []

    var_baseline = one_step_results.get('VAR', (np.nan,))[0] \
        if 'VAR' in one_step_results else np.nan
    bvar_baseline = one_step_results.get('BVAR', (np.nan,))[0] \
        if 'BVAR' in one_step_results else np.nan

    for N in sample_sizes:
        N_actual = min(N, data['X_train'].shape[0])

        sub_data = {
            'X_train': data['X_train'][:N_actual],
            'Y_train': data['Y_train'][:N_actual],
            'X_val': data['X_val'],
            'Y_val': data['Y_val'],
            'X_test': data['X_test'],
            'Y_test': data['Y_test'],
        }

        ts0 = time.perf_counter()
        sub_model, _ = train_model(
            data=sub_data, device=device,
            epochs=min(cfg['training']['epochs'], max(20, N // 10)),
            batch_size=min(cfg['training']['batch_size'], max(16, N // 10)),
            lr=cfg['training']['learning_rate'],
            checkpoint_dir=cfg['paths']['checkpoints'] / f'subset_{N}',
            patience=5,
            silent=True,
            compile_model=cfg['training']['compile'],
        )
        elapsed = time.perf_counter() - ts0
        times_by_n.append(elapsed)

        sub_mse, _ = evaluate_one_step_mse(sub_model, data, stats, device=device)
        tf_mse_by_n.append(sub_mse)

        del sub_model, sub_data
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    del data  # Free main dataset after scaling experiment
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    ss_results = {
        'sample_sizes': sample_sizes,
        'transformer_mse': tf_mse_by_n,
        'var_mse': [var_baseline] * len(sample_sizes),
        'bvar_mse': [bvar_baseline] * len(sample_sizes),
        'times': times_by_n,
    }
    with open(cfg['paths']['cache'] / 'sample_size_results.pkl', 'wb') as f:
        pickle.dump(ss_results, f)

    t6 = time.perf_counter()

    runtime_stats = {
        'data_loading': time.perf_counter() - t1,
        'training': time.perf_counter() - t2,
        'benchmarks': time.perf_counter() - t3,
        'irf_eval': time.perf_counter() - t4,
        'sample_scaling': time.perf_counter() - t5,
        'outputs': time.perf_counter() - t6,
        'total': time.perf_counter() - total_t0,
    }

    results_dict = {
        'one_step_results': one_step_results,
        'multistep_results': multistep_results,
        'irf_summary': irf_summary,
        'yonly_results': yonly_results,
        'sample_size_results': ss_results,
        'config': cfg,
        'runtime_stats': runtime_stats,
    }
    results_path = cfg['paths']['cache'].parent / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(to_jsonable(results_dict), f, indent=2)

    plot_trajectory_overlay(
        data, stats, model,
        save_path=cfg['paths']['figures'] / 'fig1_trajectory_overlay.png',
        device=device,
    )

    plot_reduced_form_trajectory_overlay(
        data, stats, model,
        save_path=cfg['paths']['figures'] / 'fig2_model_trajectory_overlay.png',
        device=device,
    )

    plot_irf_grid(
        irf_summary,
        save_path=cfg['paths']['figures'] / 'fig3_irf_grid.png',
    )

    irf_paths = collect_irf_paths(model, data, stats, sim_idx=0, horizon=20, device=device)
    plot_irf_paths(
        irf_paths,
        save_path=cfg['paths']['figures'] / 'fig3b_irf_paths.png',
    )

    var_mse_flat = [var_baseline] * len(sample_sizes)
    bvar_mse_flat = [bvar_baseline] * len(sample_sizes)
    plot_learning_curve(
        sample_sizes, tf_mse_by_n, var_mse_flat, bvar_mse_flat,
        save_path=cfg['paths']['figures'] / 'fig4_learning_curve.png',
    )

    plot_forecast_horizon(
        multistep_results,
        save_path=cfg['paths']['figures'] / 'fig5_forecast_horizon.png',
    )

    total_time = time.perf_counter() - total_t0
    tf_mse = one_step_results.get('Transformer', (np.nan,))[0]
    var_mse = one_step_results.get('VAR', (np.nan,))[0]

    stages = f"Data:{runtime_stats['data_loading']:.0f}s Train:{runtime_stats['training']:.0f}s Bench:{runtime_stats['benchmarks']:.0f}s IRF:{runtime_stats['irf_eval']:.0f}s"
    print(f"[{time.strftime('%H:%M:%S')}] Complete | {total_time:.0f}s | {stages} | TF:{tf_mse:.2e} VAR:{var_mse:.2e}")


if __name__ == '__main__':
    main()
