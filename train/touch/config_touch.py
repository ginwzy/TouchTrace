model_config = {
    "pad": -999999.0,
    "max_steps": 128,
    "components": 5,
    "input_dims": 5,
    "output_dims": 3,
    "min_delay_ms": 2.0,
    # Training data: drop micro-movements below this (px). Reduces ~40% sub-3px steps.
    "min_step_px": 3.0,
    # Gaussian noise on (dx_prev, dy_prev, dt_prev) during training — simulates inference drift.
    "input_noise_std": [1.5, 1.5, 3.0],
    # sample_weight = 1 + step_weight * |step| + dist_weight * |remaining_dist|
    "loss_step_weight": 0.15,
    "loss_dist_weight": 0.002,
    # Inverse-frequency multiplier on whole swipes by angle bucket (H/D/V), clipped.
    # With geom_aug, this is applied AFTER the D4 transform (V↔H swap); without it,
    # it is applied to the original remaining-distance vector.
    "angle_h_deg": 20.0,
    "angle_v_deg": 60.0,
    "angle_weight_max": 8.0,
    # Random D4 transform (rotations + reflections) on (dx, dy) each train sample.
    "geom_aug": True,
    # Cap per-timestep MDN NLL so a collapsed scale cannot NaN the run.
    "nll_clip": 80.0,
    # Gradient clip: LSTM+MDN can spike even when the loss is still finite.
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
    "weights": "model.h5",
    "weights_lite": "model_lite.h5",
    "onnx_model": "touch.onnx",
    "data": "touch_data.jsonl",
}
