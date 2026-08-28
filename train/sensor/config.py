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
    "validation_split": 0.1,
    "early_stopping_patience": 40,
    "reduce_lr_patience": 10,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-5,
    "remaining_frame": True,
    # Generation: 0 = mixture mean, 1 = full MDN sample. Interpolates between.
    "mdn_temp": 0.2,
    # Train: teacher-force warmup, then mix model imu_prev in with probability p.
    # Y is ΔIMU (current − previous); SS feeds sampled deltas integrated back to abs IMU.
    "ss_warmup_epochs": 20,
    "ss_ramp_epochs": 50,
    "ss_max": 1.0,
    "ss_temp": 0.2,
    "weights": "sensor_model.h5",
    "weights_lite": "sensor_model_lite.h5",
    "onnx_model": "sensor.onnx",
    "data": "sensor_data.jsonl",
    "norm": "sensor_norm.json",
}
