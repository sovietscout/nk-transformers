"""
run.py — Main entry point. Orchestrates full pipeline:
  1. Load data
  2. Train Transformer
  3. Run benchmarks
  4. Evaluate
  5. Sample size experiment
  6. Generate figures & tables
"""

import os
import time
import pickle
import argparse
import json
import numpy as np
import torch

from src.simulator import load_and_prepare
from src.train import train_model
from src.evaluate import (
    evaluate_one_step_mse,
    evaluate_var_one_step,
    evaluate_bvar_one_step,
    evaluate_multistep_mse,
    evaluate_var_multistep,
    evaluate_bvar_multistep,
    evaluate_irf_accuracy,
)
from src.plots import (
    plot_trajectory_overlay,
    plot_irf_grid,
    plot_learning_curve,
    plot_forecast_horizon,
    print_table1,
    print_table2,
    print_table3,
)


def parse_args():
    p = argparse.ArgumentParser(description='NK Transformer — Replication + Extensions')
    p.add_argument('--cache', type=str, default='./results/cache',
                   help='Cache directory for preprocessed data')
    p.add_argument('--ckpt', type=str, default='./results/checkpoints',
                   help='Checkpoint directory')
    p.add_argument('--figures', type=str, default='./results/figures',
                   help='Output directory for figures')
    p.add_argument('--device', type=str, default='cuda',
                   help='Device: cuda or cpu')
    p.add_argument('--epochs', type=int, default=100, help='Training epochs')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--skip-train', action='store_true',
                   help='Skip training, load existing checkpoint')
    p.add_argument('--skip-benchmarks', action='store_true',
                   help='Skip VAR/BVAR benchmarks (slow)')
    p.add_argument('--n-irf', type=int, default=50,
                   help='Number of simulations for IRF evaluation')
    p.add_argument('--sample-sizes', type=str,
                   default='100,300,1000,3000,10000,30000,50000',
                   help='Comma-separated N values for sample-size experiment')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    # ──────────────────────────────────────────────
    # 1. Load and normalize data
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading and normalizing data")
    print("=" * 60)

    data, stats = load_and_prepare(args.cache)

    print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, "
          f"Test: {data['X_test'].shape}")

    # ──────────────────────────────────────────────
    # 2. Train Transformer
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Training Transformer")
    print("=" * 60)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU.")

    if not args.skip_train:
        model, history = train_model(
            data=data,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=args.ckpt,
        )
        with open(os.path.join(args.ckpt, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
    else:
        from src.model import NKTransformer
        model = NKTransformer(d_model=64, n_heads=4, n_layers=3, ff_dim=256, dropout=0.1)
        ckpt_path = os.path.join(args.ckpt, 'best_model.pt')
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        model.eval()
        print(f"Loaded checkpoint from {ckpt_path}")

    # ──────────────────────────────────────────────
    # 3. Benchmarks (VAR, BVAR)
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Running benchmarks")
    print("=" * 60)

    one_step_results = {}
    multistep_results = {}

    # --- Transformer one-step MSE ---
    print("\nTransformer one-step MSE...")
    tf_one_mse, tf_one_pervar = evaluate_one_step_mse(model, data, stats, device=device)
    one_step_results['Transformer'] = (tf_one_mse, tf_one_pervar)
    print(f"  Overall: {tf_one_mse:.6f}, per-var: {tf_one_pervar}")

    # --- Transformer multi-step MSE ---
    print("\nTransformer multi-step MSE...")
    tf_ms_mse = evaluate_multistep_mse(model, data, stats, device=device)
    multistep_results['Transformer'] = tf_ms_mse
    for h, (mse, pv) in tf_ms_mse.items():
        print(f"  h={h:2d}: {mse:.6f}")

    if not args.skip_benchmarks:
        # --- OLS VAR ---
        print("\nOLS VAR one-step MSE...")
        try:
            var_one_mse, var_one_pervar = evaluate_var_one_step(data, stats)
            one_step_results['VAR'] = (var_one_mse, var_one_pervar)
            print(f"  Overall: {var_one_mse:.6f}, per-var: {var_one_pervar}")
        except Exception as e:
            print(f"  VAR one-step failed: {e}")

        print("OLS VAR multi-step MSE...")
        try:
            var_ms_mse = evaluate_var_multistep(data, stats)
            multistep_results['VAR'] = var_ms_mse
        except Exception as e:
            print(f"  VAR multi-step failed: {e}")

        # --- BVAR ---
        print("\nBVAR one-step MSE...")
        try:
            bvar_one_mse, bvar_one_pervar = evaluate_bvar_one_step(data, stats)
            one_step_results['BVAR'] = (bvar_one_mse, bvar_one_pervar)
            print(f"  Overall: {bvar_one_mse:.6f}, per-var: {bvar_one_pervar}")
        except Exception as e:
            print(f"  BVAR one-step failed: {e}")

        print("BVAR multi-step MSE...")
        try:
            bvar_ms_mse = evaluate_bvar_multistep(data, stats)
            multistep_results['BVAR'] = bvar_ms_mse
        except Exception as e:
            print(f"  BVAR multi-step failed: {e}")

    # ──────────────────────────────────────────────
    # 4. IRF evaluation
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: IRF evaluation")
    print("=" * 60)
    irf_summary = evaluate_irf_accuracy(model, data, stats,
                                        n_sims=args.n_irf, device=device)

    # ──────────────────────────────────────────────
    # 5. Sample size experiment
    # ──────────────────────────────────────────────
    # Only the Transformer benefits from more training simulations — VAR and BVAR
    # are per-simulation estimators (each test sim gets its own VAR fit on its
    # own 150 time periods). So we show VAR/BVAR as flat baselines at their
    # full-data MSE, and the Transformer's learning curve vs N.

    print("\n" + "=" * 60)
    print("STEP 5: Sample size experiment")
    print("=" * 60)
    print("Note: VAR/BVAR are per-simulation estimators, not improved by larger N.")
    print("      Their values here are flat baselines from full evaluation.")

    sample_sizes = [int(x.strip()) for x in args.sample_sizes.split(',')]
    tf_mse_by_n = []
    times_by_n = []

    # Get VAR/BVAR baselines (N-independent, from full evaluation above)
    var_baseline = one_step_results.get('VAR', (np.nan,))[0] \
        if 'VAR' in one_step_results else np.nan
    bvar_baseline = one_step_results.get('BVAR', (np.nan,))[0] \
        if 'BVAR' in one_step_results else np.nan

    for N in sample_sizes:
        print(f"\n--- N = {N} ---")
        N_actual = min(N, data['X_train'].shape[0])
        X_sub = data['X_train'][:N_actual]
        Y_sub = data['Y_train'][:N_actual]

        sub_data = {
            'X_train': X_sub, 'Y_train': Y_sub,
            'X_val': data['X_val'], 'Y_val': data['Y_val'],
            'X_test': data['X_test'], 'Y_test': data['Y_test'],
        }

        t0 = time.time()
        sub_model, _ = train_model(
            data=sub_data, device=device,
            epochs=min(args.epochs, max(20, N // 10)),
            batch_size=min(args.batch_size, max(16, N // 10)),
            lr=args.lr,
            checkpoint_dir=os.path.join(args.ckpt, f'subset_{N}'),
            patience=5,
        )
        elapsed = time.time() - t0
        times_by_n.append(elapsed)

        sub_mse, _ = evaluate_one_step_mse(sub_model, data, stats, device=device)
        tf_mse_by_n.append(sub_mse)
        print(f"  Transformer: {sub_mse:.6f} (VAR baseline: {var_baseline:.6f}, "
              f"BVAR baseline: {bvar_baseline:.6f}), Time: {elapsed:.1f}s")

    # Save sample size results
    ss_results = {
        'sample_sizes': sample_sizes,
        'transformer_mse': tf_mse_by_n,
        'var_mse': [var_baseline] * len(sample_sizes),
        'bvar_mse': [bvar_baseline] * len(sample_sizes),
        'times': times_by_n,
    }
    with open(os.path.join(args.cache, 'sample_size_results.pkl'), 'wb') as f:
        pickle.dump(ss_results, f)

    # ──────────────────────────────────────────────
    # 6. Produce tables and figures
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Generating tables and figures")
    print("=" * 60)

    # Tables
    print_table1(one_step_results)
    print_table2(multistep_results)
    print_table3(irf_summary)

    # Save results.json to run folder root
    results_dict = {
        'one_step_results': {k: (float(v[0]), v[1].tolist() if hasattr(v[1], 'tolist') else v[1])
                             for k, v in one_step_results.items()},
        'multistep_results': {k: {h: (float(v[0]), v[1].tolist() if hasattr(v[1], 'tolist') else v[1])
                                 for h, v in v.items()}
                             for k, v in multistep_results.items()},
        'irf_summary': irf_summary,
        'sample_size_results': {str(k): v for k, v in ss_results.items()},
        'training_args': vars(args),
    }
    with open(os.path.join(args.cache, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Figure 1: Trajectory overlay
    plot_trajectory_overlay(
        data, stats, model,
        save_path=os.path.join(args.figures, 'fig1_trajectory_overlay.png'),
        device=device,
    )

    # Figure 2 removed — requires VAR/BVAR trajectory computation not implemented

    # Figure 3: IRF grid
    plot_irf_grid(
        irf_summary,
        save_path=os.path.join(args.figures, 'fig3_irf_grid.png'),
    )

    # Figure 4: Learning curve — only Transformer varies with N;
    # VAR/BVAR are per-simulation estimators and shown as flat baselines.
    var_mse_flat = [var_baseline] * len(sample_sizes)
    bvar_mse_flat = [bvar_baseline] * len(sample_sizes)
    plot_learning_curve(
        sample_sizes, tf_mse_by_n, var_mse_flat, bvar_mse_flat,
        save_path=os.path.join(args.figures, 'fig4_learning_curve.png'),
    )

    # Figure 5: Forecast horizon MSE
    plot_forecast_horizon(
        multistep_results,
        save_path=os.path.join(args.figures, 'fig5_forecast_horizon.png'),
    )

    print("\n✅ Done! All results saved.")
    print(f"   Figures: {args.figures}/")
    print(f"   Checkpoints: {args.ckpt}/")
    print(f"   Cache: {args.cache}/")


if __name__ == '__main__':
    main()
