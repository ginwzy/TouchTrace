model_config = {
    "pad": -999999.0,
    "max_steps": 128,
    "components": 5,
    # imu_prev(6) + tan/nrm/dt/rem(4) + condition one-hot(3)
    "input_dims": 13,
    "output_dims": 6,
    "min_step_px": 3.0,
    # Noise on z-scored previous IMU only (first 6 input channels).
    "input_noise_std": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    "nll_clip": 80.0,
    "clipnorm": 5.0,
    "epochs": 250,
    "batch_size": 256,
    "lstm_units": 128,
    "lstm_units_lite": 64,
    "learning_rate": 0.0005,
    # Keep users disjoint across training, checkpoint selection, and final test.
    "split_by_user": True,
    "split_seed": 42,
    "validation_split": 0.1,
    "test_split": 0.1,
    "split_manifest": "sensor_split.json",
    # Save rollout candidates for offline distribution-aware selection.
    "rollout_checkpoint_interval": 5,
    "early_stopping_patience": 40,
    "reduce_lr_patience": 10,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-5,
    "remaining_frame": True,
    # Generation: 0 = mixture mean, 1 = full MDN sample. Interpolates between.
    "mdn_temp": 0.2,
    # Train: teacher-force warmup, then mix model imu_prev in with probability p.
    # SS unrolls closed-loop steps and retargets Y toward the human next state.
    "ss_warmup_epochs": 20,
    "ss_ramp_epochs": 50,
    "ss_max": 1.0,
    "ss_temp": 0.2,
    "ss_unroll_hops": 4,
    # correction = legacy bounded controller target; natural leaves MDN labels
    # on the human delta; hybrid adds a small mean-only correction Huber term.
    "ss_target_mode": "correction",
    "ss_hybrid_weight": 0.05,
    # Retarget corrections are z-scored natural deltas. Limit only synthetic
    # correction labels so off-manifold states cannot inflate the MDN scales.
    "ss_target_clip_z": 4.0,
    "ss_checkpoint_probs": [0.2, 0.5, 0.8, 1.0],
    # Deterministic mixture-mean rollout is a divergence guard, not a selector.
    "ar_eval_paths": 32,
    "ar_eval_steps": 32,
    "ar_stability_max_mae_z": 2.0,
    "weights": "sensor_model.h5",
    "weights_lite": "sensor_model_lite.h5",
    "onnx_model": "sensor.onnx",
    "data": "sensor_data.jsonl",
    "norm": "sensor_norm.json",
}
