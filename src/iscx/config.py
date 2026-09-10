"""Feature list and MLP configuration shared by the ISCX classifier scripts."""

RANDOM_STATE = 42

FEATURE_COLS = [
    "duration", "total_fiat", "total_biat", "min_fiat", "min_biat",
    "max_fiat", "max_biat", "mean_fiat", "mean_biat",
    "flowPktsPerSecond", "flowBytesPerSecond",
    "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle",
]

MLP_KWARGS = dict(
    hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
    alpha=1e-4, batch_size=256, learning_rate="adaptive",
    learning_rate_init=1e-3, max_iter=200, random_state=RANDOM_STATE,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
)
