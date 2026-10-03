"""Model zoo. All models receive the SAME windows (sequence models) or a tabular summary of the SAME windows (classical models)
and predict the SAME targets: capped RUL (normalised to [0,1] by the cap) and, where they have a second head, the 3-class stage."""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from . import config as C
from .data import window_summary, stage_from_rul_min

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks


def set_determinism(seed):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@tf.keras.utils.register_keras_serializable(package="bp")
class TemporalAttention(layers.Layer):
    """alpha_t = softmax_t(u^T tanh(W h_t + b));  context = sum_t alpha_t h_t.  Returns (context, alpha)."""
    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(name="W", shape=(d, d), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(d,), initializer="zeros")
        self.u = self.add_weight(name="u", shape=(d, 1), initializer="glorot_uniform")

    def call(self, x):
        score = tf.tensordot(tf.tanh(tf.tensordot(x, self.W, 1) + self.b), self.u, 1)     # (B, T, 1)
        alpha = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(x * alpha, axis=1), alpha


NN_KINDS = {  # key -> (label, dual_head)
    "cnn": ("CNN", False), "lstm": ("LSTM", False), "cnn_lstm": ("CNN + LSTM", False),
    "cnn_bilstm": ("CNN + BiLSTM", False), "cnn_bilstm_attn": ("CNN + BiLSTM + Attention (single head)", False),
    "cnn_bilstm_dual": ("CNN + BiLSTM (dual head, no attention)", True),
    "proposed": ("PROPOSED: CNN + BiLSTM + Attention (dual head)", True),
}


def build_nn(kind, n_feats, window=C.WINDOW):
    dual = NN_KINDS[kind][1]
    inp = layers.Input(shape=(window, n_feats), name="sequence_input")
    x = inp
    if kind in ("cnn", "cnn_lstm", "cnn_bilstm", "cnn_bilstm_attn", "cnn_bilstm_dual", "proposed"):
        x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1")(x)
        x = layers.BatchNormalization(name="bn1")(x); x = layers.Dropout(0.2, name="drop1")(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv2")(x)
        x = layers.BatchNormalization(name="bn2")(x); x = layers.Dropout(0.2, name="drop2")(x)
    if kind == "cnn":
        z = layers.GlobalAveragePooling1D(name="pool")(x)
    elif kind == "lstm":
        z = layers.LSTM(64, name="lstm")(x)
    elif kind == "cnn_lstm":
        z = layers.LSTM(64, name="lstm")(x)
    else:
        h = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm")(x)
        h = layers.Dropout(0.2, name="drop_bilstm")(h)
        if kind in ("cnn_bilstm_attn", "proposed"):
            z, _ = TemporalAttention(name="temporal_attention")(h)
        else:
            z = layers.GlobalAveragePooling1D(name="pool")(h)
    s = layers.Dense(64, activation="relu", name="shared_dense")(z)
    s = layers.Dropout(0.2, name="drop_shared")(s)
    r = layers.Dense(32, activation="relu", name="rul_dense")(s)
    rul = layers.Dense(1, activation="sigmoid", name="rul_output")(r)
    if not dual:
        m = Model(inp, {"rul_output": rul}, name=kind)
        m.compile(tf.keras.optimizers.Adam(C.LR), {"rul_output": tf.keras.losses.Huber(C.HUBER_DELTA)}, metrics={"rul_output": "mae"})
        return m
    k = layers.Dense(32, activation="relu", name="risk_dense")(s)
    risk = layers.Dense(3, activation="softmax", name="risk_output")(k)
    m = Model(inp, {"rul_output": rul, "risk_output": risk}, name=kind)
    m.compile(tf.keras.optimizers.Adam(C.LR), {"rul_output": tf.keras.losses.Huber(C.HUBER_DELTA), "risk_output": "sparse_categorical_crossentropy"},
              loss_weights=C.LOSS_WEIGHTS, metrics={"rul_output": "mae", "risk_output": "accuracy"})
    return m


def stage_class_weights(stage):
    cls = np.array([0, 1, 2])
    present = np.unique(stage)
    w = compute_class_weight("balanced", classes=present, y=stage)
    d = {c: 1.0 for c in cls}; d.update({c: wi for c, wi in zip(present, w)})
    return np.array([d[s] for s in stage], dtype="float32")


def fit_nn(kind, Xtr, meta_tr, seed, Xva=None, meta_va=None, epochs=None):
    """Train one network. With a validation set: early stopping (restore best); returns best epoch. Without: fixed `epochs`."""
    set_determinism(seed)
    m = build_nn(kind, Xtr.shape[-1], Xtr.shape[1])
    dual = NN_KINDS[kind][1]
    ytr = {"rul_output": meta_tr.y_rul.values.astype("float32")}
    sw = None
    if dual:
        ytr["risk_output"] = meta_tr.stage.values.astype("int32")
        sw = {"rul_output": np.ones(len(Xtr), "float32"), "risk_output": stage_class_weights(meta_tr.stage.values)}
    kw = dict(batch_size=C.BATCH, verbose=0, sample_weight=sw)
    if Xva is not None:
        yva = {"rul_output": meta_va.y_rul.values.astype("float32")}
        if dual:
            yva["risk_output"] = meta_va.stage.values.astype("int32")
        cb = [callbacks.EarlyStopping(monitor="val_loss", patience=C.PATIENCE, restore_best_weights=True),
              callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]
        h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=C.MAX_EPOCHS, callbacks=cb, **kw)
        best = int(np.argmin(h.history["val_loss"])) + 1
    else:
        h = m.fit(Xtr, ytr, epochs=int(epochs), **kw); best = int(epochs)
    return m, best, h.history


def predict_nn(m, X, kind):
    p = m.predict(X, batch_size=512, verbose=0)
    y = np.clip(np.asarray(p["rul_output"]).ravel(), 0, 1)
    prob = np.asarray(p["risk_output"]) if NN_KINDS[kind][1] else None
    return {"pred_min": y * C.RUL_CAP_MIN, "prob": prob}


# ------------------------------------------------------------------ classical models
CLASSICAL = ["Constant", "Ridge", "Random Forest", "Gradient Boosting"]


class Classical:
    def __init__(self, name, seed):
        self.name, self.seed = name, seed

    def fit(self, X, meta):
        T = window_summary(X); y = meta.y_rul.values; s = meta.stage.values
        n = self.name
        if n == "Constant":
            self.mu, self.maj = float(y.mean()), int(np.bincount(s, minlength=3).argmax()); return self
        if n == "Ridge":
            self.r = Ridge(alpha=1.0).fit(T, y); self.c = LogisticRegression(max_iter=3000, class_weight="balanced").fit(T, s)
        elif n == "Random Forest":
            self.r = RandomForestRegressor(200, min_samples_leaf=5, n_jobs=-1, random_state=self.seed).fit(T, y)
            self.c = RandomForestClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=self.seed, class_weight="balanced").fit(T, s)
        elif n == "Gradient Boosting":
            self.r = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=self.seed).fit(T, y)
            self.c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=self.seed, class_weight="balanced").fit(T, s)
        self.classes_ = self.c.classes_
        return self

    def predict(self, X):
        if self.name == "Constant":
            prob = np.zeros((len(X), 3)); prob[:, self.maj] = 1
            return {"pred_min": np.full(len(X), self.mu * C.RUL_CAP_MIN), "prob": prob}
        T = window_summary(X)
        y = np.clip(self.r.predict(T), 0, 1)
        pr = self.c.predict_proba(T); prob = np.zeros((len(X), 3)); prob[:, self.classes_] = pr
        return {"pred_min": y * C.RUL_CAP_MIN, "prob": prob}


def layer_table(model):
    """Layer-by-layer table (input shape, output shape, parameters) generated from the built Keras model."""
    rows = []
    for l in model.layers:
        try:
            ish = l.input.shape if not isinstance(l.input, (list, tuple, dict)) else [t.shape for t in l.input]
        except Exception:
            ish = "-"
        try:
            osh = l.output.shape if not isinstance(l.output, (list, tuple, dict)) else [t.shape for t in l.output]
        except Exception:
            osh = "-"
        rows.append({"layer": l.name, "type": l.__class__.__name__, "input_shape": str(ish), "output_shape": str(osh), "params": l.count_params()})
    return pd.DataFrame(rows)
