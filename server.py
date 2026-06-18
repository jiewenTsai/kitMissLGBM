"""Shiny for Python server — reactive logic, extended tasks, render/download."""
import asyncio
import queue
from pathlib import Path

from shiny import reactive, render, req, ui

import backend.export as export
import backend.io_data as io_data
import backend.plots as plots
import backend.shap_analysis as shap_analysis
import backend.train as train
from ui_helpers import progress_bar_ui, status_alert


def server(input, output, session):

    # ── Session state ─────────────────────────────────────────────────────────
    _train_result: reactive.Value[dict | None] = reactive.Value(None)
    _train_status: reactive.Value[str] = reactive.Value("idle")
    _shap_status:  reactive.Value[str] = reactive.Value("idle")
    _shap_error_msg: reactive.Value[str] = reactive.Value("")

    # Thread-safe queues for progress reporting from background threads.
    _train_q: queue.SimpleQueue = queue.SimpleQueue()
    _shap_q:  queue.SimpleQueue = queue.SimpleQueue()

    # Progress state: {"phase": "idle"|"flaml"|"oof"|"shap", "done": int, "total": int, "msg": str}
    _train_prog: reactive.Value[dict] = reactive.Value(
        {"phase": "idle", "done": 0, "total": 0, "msg": ""}
    )
    _shap_prog: reactive.Value[dict] = reactive.Value(
        {"phase": "idle", "done": 0, "total": 0, "msg": ""}
    )

    # ── Progress polling (0.5 s interval) ─────────────────────────────────────
    @reactive.effect
    def _poll_train_progress():
        reactive.invalidate_later(0.5)
        latest = None
        while True:
            try:
                latest = _train_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            _train_prog.set(latest)

    @reactive.effect
    def _poll_shap_progress():
        reactive.invalidate_later(0.5)
        latest = None
        while True:
            try:
                latest = _shap_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            _shap_prog.set(latest)

    # ── Task: model training ──────────────────────────────────────────────────

    @ui.bind_task_button(button_id="btn_train")
    @reactive.extended_task
    async def _train_task(X, y, wave_label, time_budget, n_splits, seed):
        loop = asyncio.get_event_loop()

        def _run():
            # Phase 1: FLAML hyperparameter search (duration unknown → indeterminate bar)
            _train_q.put({
                "phase": "flaml", "done": 0, "total": 0,
                "msg": f"FLAML 搜尋最佳參數中（時間預算 {time_budget}s）...",
            })
            model, cfg = train.run_flaml_train(
                X, y,
                time_budget=time_budget,
                n_splits=n_splits,
                seed=seed,
            )

            # Phase 2: OOF metrics (determinate bar — n_seeds × n_splits folds)
            def _oof_cb(done: int, total: int) -> None:
                _train_q.put({
                    "phase": "oof", "done": done, "total": total,
                    "msg": f"OOF 評估：{done}/{total} 折",
                })

            metrics_df = train.compute_oof_metrics(
                model, X, y, n_splits=n_splits, progress_cb=_oof_cb
            )
            _train_q.put({"phase": "idle", "done": 0, "total": 0, "msg": ""})
            return {
                "model": model,
                "X": X,
                "y": y,
                "wave_label": wave_label,
                "metrics_df": metrics_df,
                "best_config": cfg,
            }

        return await loop.run_in_executor(None, _run)

    @reactive.effect
    @reactive.event(input.btn_train)
    def _invoke_train():
        # Drain stale queue items from any previous run
        while True:
            try:
                _train_q.get_nowait()
            except queue.Empty:
                break
        _train_prog.set({"phase": "idle", "done": 0, "total": 0, "msg": ""})

        f = input.data_csv()
        if f is None:
            _train_status.set("error")
            return
        _train_status.set("running")
        csv_bytes = Path(f[0]["datapath"]).read_bytes()
        df, X, y = io_data.load_csv_bytes(csv_bytes, input.target_column())
        _train_task.invoke(
            X, y,
            input.wave_label(),
            input.time_budget(),
            input.n_splits(),
            input.seed(),
        )

    @reactive.effect
    def _watch_train():
        """Store result when training completes successfully."""
        try:
            r = _train_task.result()
            _train_result.set(r)
            _train_status.set("success")
        except Exception:
            pass  # task not started, running, or failed — handled elsewhere

    # ── Train render outputs ──────────────────────────────────────────────────

    @render.ui
    def train_progress_bar():
        p = _train_prog.get()
        return progress_bar_ui(p["phase"], p["done"], p["total"], p["msg"])

    @render.ui
    def data_summary():
        f = input.data_csv()
        if f is None:
            return None
        try:
            csv_bytes = Path(f[0]["datapath"]).read_bytes()
            df, X, y = io_data.load_csv_bytes(csv_bytes, input.target_column())
            n_pos, n_total = int(y.sum()), len(y)
            return ui.div(
                ui.tags.p(f"資料維度：{n_total} 列 × {X.shape[1] + 1} 欄"),
                ui.tags.p(
                    f"流失（1）：{n_pos}（{n_pos / n_total:.1%}）　"
                    f"未流失（0）：{n_total - n_pos}（{(n_total - n_pos) / n_total:.1%}）"
                ),
                class_="alert alert-info",
                style="margin-top: 8px;",
            )
        except Exception as e:
            return status_alert("error", f"資料讀取失敗：{e}")

    @render.ui
    def train_status():
        st = _train_status.get()
        if st == "idle":
            return None
        if st == "running":
            return status_alert("running", "模型訓練進行中，訓練完成前可切換至其他分頁。")
        if st == "error":
            return status_alert("error", "訓練失敗，請確認 CSV 格式與目標變項欄名是否正確。")
        r = _train_result.get()
        if r is None:
            return None
        return status_alert(
            "success",
            f"訓練完成 | 波次：{r['wave_label']}",
        )

    @render.table
    def metrics_table():
        r = _train_task.result()
        return r["metrics_df"]

    @render.table
    def best_config_table():
        r = _train_task.result()
        cfg = r["best_config"]
        import pandas as pd
        return pd.DataFrame(
            [{"參數": k, "數值": f"{v:.6g}" if isinstance(v, float) else str(v)}
             for k, v in sorted(cfg.items())]
        )

    # ── Train downloads ───────────────────────────────────────────────────────

    @render.download(filename=lambda: f"flaml_model_{input.wave_label()}.pkl")
    def dl_model_pkl():
        r = req(_train_result.get())
        yield export.model_to_pkl_bytes(r["model"])

    @render.download(filename="oof_metrics_summary.csv")
    def dl_metrics_csv():
        r = req(_train_result.get())
        yield export.df_to_csv_bytes(r["metrics_df"])

    # ── Task: OOF SHAP ────────────────────────────────────────────────────────

    @ui.bind_task_button(button_id="btn_shap")
    @reactive.extended_task
    async def _shap_task(model, X, y, shap_seeds, shap_n_folds, top_n, cum_threshold):
        loop = asyncio.get_event_loop()

        def _run():
            total = shap_seeds * shap_n_folds

            def _shap_cb(done: int, _total: int) -> None:
                _shap_q.put({
                    "phase": "shap", "done": done, "total": total,
                    "msg": f"OOF SHAP：{done}/{total} 折",
                })

            result = shap_analysis.run_oof_shap(
                model, X, y,
                shap_seeds=shap_seeds,
                shap_n_folds=shap_n_folds,
                top_n=top_n,
                cum_threshold=cum_threshold,
                progress_cb=_shap_cb,
            )
            _shap_q.put({"phase": "idle", "done": 0, "total": 0, "msg": ""})
            return result

        return await loop.run_in_executor(None, _run)

    @reactive.effect
    @reactive.event(input.btn_shap)
    def _invoke_shap():
        while True:
            try:
                _shap_q.get_nowait()
            except queue.Empty:
                break
        _shap_prog.set({"phase": "idle", "done": 0, "total": 0, "msg": ""})

        src = input.model_source()
        _shap_error_msg.set("")

        if src == "inherit":
            r = _train_result.get()
            if r is None:
                _shap_status.set("error")
                _shap_error_msg.set("請先在「模型訓練」分頁完成訓練，或改選「上傳外部 .pkl 模型」。")
                return
            model, X, y = r["model"], r["X"], r["y"]
        else:
            pkl_f = input.model_pkl()
            csv_f = input.shap_csv()
            if pkl_f is None or csv_f is None:
                _shap_status.set("error")
                _shap_error_msg.set("請同時上傳 .pkl 模型檔與對應 CSV 後再執行。")
                return
            try:
                model = io_data.load_model_bytes(
                    Path(pkl_f[0]["datapath"]).read_bytes()
                )
                _, X, y = io_data.load_csv_bytes(
                    Path(csv_f[0]["datapath"]).read_bytes(),
                    input.shap_target_column(),
                )
            except Exception as e:
                _shap_status.set("error")
                _shap_error_msg.set(str(e))
                return

        _shap_status.set("running")
        _shap_task.invoke(
            model, X, y,
            input.shap_seeds(),
            input.shap_n_folds(),
            input.top_n(),
            input.cum_threshold(),
        )

    @reactive.effect
    def _watch_shap():
        status = _shap_task.status()
        if status == "success":
            _shap_status.set("success")
            _shap_error_msg.set("")
        elif status == "error":
            _shap_status.set("error")
            try:
                _shap_error_msg.set(str(_shap_task.error()))
            except Exception:
                _shap_error_msg.set("SHAP 計算失敗，請確認模型與資料相容。")

    @render.ui
    def shap_data_summary():
        if input.model_source() != "external":
            return None
        csv_f = input.shap_csv()
        if csv_f is None:
            return None
        try:
            csv_bytes = Path(csv_f[0]["datapath"]).read_bytes()
            df, X, y = io_data.load_csv_bytes(
                csv_bytes, input.shap_target_column()
            )
            n_pos, n_total = int(y.sum()), len(y)
            return ui.div(
                ui.tags.p(f"資料維度：{n_total} 列 × {X.shape[1] + 1} 欄"),
                ui.tags.p(
                    f"流失（1）：{n_pos}（{n_pos / n_total:.1%}）　"
                    f"未流失（0）：{n_total - n_pos}（{(n_total - n_pos) / n_total:.1%}）"
                ),
                style="margin-top: 8px;",
            )
        except Exception as e:
            return status_alert("error", f"資料讀取失敗：{e}")

    # ── SHAP render outputs ───────────────────────────────────────────────────

    @render.ui
    def shap_progress_bar():
        p = _shap_prog.get()
        return progress_bar_ui(p["phase"], p["done"], p["total"], p["msg"])

    @render.ui
    def shap_status():
        st = _shap_status.get()
        if st == "idle":
            return None
        if st == "running":
            n_total = input.shap_seeds() * input.shap_n_folds()
            return status_alert(
                "running",
                f"SHAP 計算進行中（{input.shap_seeds()} seeds × "
                f"{input.shap_n_folds()} folds = {n_total} 折），請稍候...",
            )
        if st == "error":
            msg = _shap_error_msg.get()
            if not msg:
                msg = (
                    "SHAP 計算失敗，請確認模型與資料相容（欄位須與訓練時一致），"
                    "或先在「模型訓練」分頁完成訓練。"
                )
            return status_alert("error", msg)
        try:
            r = _shap_task.result()
            n_cum = int((r["shap_df"]["cumulative_pct"] <= r["cum_threshold"]).sum())
            return status_alert(
                "success",
                f"SHAP 完成 | Top-{r['top_n']} 已顯示 | "
                f"累積 {r['cum_threshold']:.0%} 閾值：{n_cum} 個變項",
            )
        except Exception:
            return None

    @render.plot
    def plot_bar():
        r = _shap_task.result()
        return plots.plot_shap_bar_ci(r["shap_df"], r["top_n"])

    @render.plot
    def plot_beeswarm():
        r = _shap_task.result()
        top_features = r["shap_df"]["feature"].iloc[: r["top_n"]].tolist()
        return plots.plot_shap_beeswarm(r["sv_full"], r["X"], top_features, r["top_n"])

    @render.plot
    def plot_cum():
        r = _shap_task.result()
        return plots.plot_shap_cumulative(r["shap_df"], r["top_n"], r["cum_threshold"])

    @render.data_frame
    def shap_table():
        r = _shap_task.result()
        df = r["shap_df"][["rank", "feature", "mean_abs_shap", "ci_low", "ci_high"]].copy()
        df.columns = ["排名", "變項", "Mean |SHAP|", "CI 下界 (2.5%)", "CI 上界 (97.5%)"]
        for col in ["Mean |SHAP|", "CI 下界 (2.5%)", "CI 上界 (97.5%)"]:
            df[col] = df[col].round(6)
        return render.DataGrid(df, height="400px")

    # ── SHAP downloads ────────────────────────────────────────────────────────

    def _require_shap_result() -> dict:
        req(_shap_task.status() == "success")
        return _shap_task.result()

    @render.download(filename="shap_importance_all.csv")
    def dl_shap_all_csv():
        r = _require_shap_result()
        yield export.df_to_csv_bytes(r["shap_df"])

    @render.download(filename="selected_variables.csv")
    def dl_shap_sel_csv():
        r = _require_shap_result()
        sel = r["shap_df"].loc[
            r["shap_df"]["cumulative_pct"] <= r["cum_threshold"],
            ["rank", "feature"],
        ]
        yield export.df_to_csv_bytes(sel)

    @render.download(filename="shap_bar_ci.png")
    def dl_shap_bar_png():
        r = _require_shap_result()
        fig = plots.plot_shap_bar_ci(r["shap_df"], r["top_n"])
        yield export.fig_to_png_bytes(fig)

    @render.download(filename="shap_beeswarm.png")
    def dl_shap_beeswarm_png():
        r = _require_shap_result()
        top_features = r["shap_df"]["feature"].iloc[: r["top_n"]].tolist()
        fig = plots.plot_shap_beeswarm(r["sv_full"], r["X"], top_features, r["top_n"])
        yield export.fig_to_png_bytes(fig)

    @render.download(filename="shap_cumulative.png")
    def dl_shap_cum_png():
        r = _require_shap_result()
        fig = plots.plot_shap_cumulative(r["shap_df"], r["top_n"], r["cum_threshold"])
        yield export.fig_to_png_bytes(fig)

    @render.download(filename="shap_all_outputs.zip")
    def dl_shap_zip():
        r = _require_shap_result()
        shap_df = r["shap_df"]
        sel_df = shap_df.loc[
            shap_df["cumulative_pct"] <= r["cum_threshold"], ["rank", "feature"]
        ]
        top_features = shap_df["feature"].iloc[: r["top_n"]].tolist()
        fig_bar = plots.plot_shap_bar_ci(shap_df, r["top_n"])
        fig_bee = plots.plot_shap_beeswarm(r["sv_full"], r["X"], top_features, r["top_n"])
        fig_cum = plots.plot_shap_cumulative(shap_df, r["top_n"], r["cum_threshold"])
        yield export.make_shap_zip(shap_df, sel_df, fig_bar, fig_bee, fig_cum)
