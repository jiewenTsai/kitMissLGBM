"""Shiny for Python UI — three-tab layout (模型訓練 / SHAP 分析 / 重要名詞)."""
from shiny import ui

from ui_glossary import tab_glossary
from ui_helpers import download_hint, goal_panel, kit_header, make_theme, progress_bar_ui  # noqa: F401

# ── Citation strings ──────────────────────────────────────────────────────────
_CITE_FLAML = (
    'Wang, C., Wu, Q., Weimer, M., &amp; Zhu, E. (2021). FLAML: A fast and lightweight AutoML library. '
    '<em>Proceedings of Machine Learning and Systems</em>, <em>3</em>, 434–447.'
)
_CITE_LGBM = (
    'Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. '
    '<em>Advances in Neural Information Processing Systems</em>, <em>30</em>, 3146–3154.'
)
_CITE_SHAP = (
    'Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model predictions. '
    '<em>Advances in Neural Information Processing Systems</em>, <em>30</em>, 4765–4774.'
)

# ── Tab builders ─────────────────────────────────────────────────────────────

def _tab_train() -> ui.Tag:
    return ui.nav_panel(
        "模型訓練",
        ui.br(),

        goal_panel(
            objective="上傳 CSV 資料，以 FLAML 自動調參訓練 LightGBM 流失預測模型，並以 OOF 方式評估模型效能。",
            tool="FLAML AutoML（BlendSearch 超參數搜尋）、LightGBM（GBDT 分類器）、StratifiedKFold OOF 評估",
            steps=[
                "上傳含目標變項（預設 attrition）的 CSV 檔",
                "確認波次標籤與訓練參數（時間預算、CV folds、seed）",
                "按「開始訓練」—— 訓練期間可切換其他分頁",
                "訓練完成後檢視 OOF 指標摘要表",
                "下載模型 .pkl 與指標 CSV（可供後續 SHAP 分析使用）",
            ],
            result=(
                "OOF 模型評估摘要（AUC、AUPRC、F1、Recall、Brier score）"
                "，以及可下載的訓練模型檔（.pkl）。"
            ),
            citation=f"{_CITE_FLAML}<br>{_CITE_LGBM}",
        ),

        # ── 資料上傳與基本設定 ─────────────────────────────────
        ui.h4("資料輸入"),
        ui.input_file(
            "data_csv",
            "上傳資料檔（CSV，單檔上限 200 MB）",
            accept=[".csv"],
            multiple=False,
        ),
        ui.output_ui("data_summary"),
        ui.hr(),

        ui.h4("訓練參數"),
        ui.layout_columns(
            ui.input_text("target_column", "目標變項欄名", value="attrition"),
            ui.input_text("wave_label",    "波次標籤（用於檔名）", value="M36"),
            col_widths=[4, 4],
        ),
        ui.layout_columns(
            ui.input_slider("time_budget", "時間預算（秒）", min=60, max=600, step=60, value=300),
            ui.input_slider("n_splits",    "CV folds",      min=3,  max=10, step=1,  value=5),
            ui.input_numeric("seed", "Random seed", value=20260612),
            col_widths=[4, 4, 4],
        ),
        ui.input_task_button("btn_train", "開始訓練"),
        ui.output_ui("train_progress_bar"),
        ui.output_ui("train_status"),
        ui.hr(),

        # ── 訓練結果 ───────────────────────────────────────────
        ui.h4("OOF 模型評估摘要"),
        ui.tags.p(
            class_="text-muted",
            *[
                "10 seeds × CV folds 的 OOF 指標（平均 ± SD）。",
                ui.tags.br(),
                "「判斷標準 / 備註」欄提供各指標的參考解讀。",
            ],
        ),
        ui.output_table("metrics_table"),
        ui.br(),

        ui.h4("FLAML 最佳超參數"),
        ui.tags.p(
            "FLAML 在時間預算內搜尋到的最佳 LightGBM 參數組合。",
            class_="text-muted",
        ),
        ui.output_table("best_config_table"),
        ui.br(),

        # ── 匯出 ───────────────────────────────────────────────
        ui.h4("匯出"),
        ui.download_button("dl_model_pkl",  "下載模型 .pkl",   class_="btn-primary me-2"),
        ui.download_button("dl_metrics_csv", "下載指標 CSV",   class_="btn-outline-primary"),
        download_hint(),
    )


def _tab_shap() -> ui.Tag:
    return ui.nav_panel(
        "SHAP 分析",
        ui.br(),

        goal_panel(
            objective=(
                "以 OOF SHAP 識別對流失風險貢獻最大的變項，"
                "提供重要性排序（含 95% CI）、效果方向（Beeswarm）及累積比例圖，"
                "供研究團隊規劃追蹤策略。"
            ),
            tool="SHAP TreeExplainer、OOF SHAP（50 折均值 + empirical percentile CI）",
            steps=[
                "選擇模型來源：繼承「模型訓練」結果，或上傳外部 .pkl 模型（需另上傳對應 CSV）",
                "設定 SHAP 計算參數（seeds、folds、Top-N、累積閾值）",
                "按「執行 SHAP」—— 計算期間可切換其他分頁",
                "檢視三張圖表與重要性表格",
                "逐一下載圖片 / CSV，或一鍵下載全部 ZIP",
            ],
            result=(
                "Top-N 重要性長條圖（OOF SHAP + 95% CI）、"
                "Beeswarm 效果方向圖、累積 SHAP 比例圖；"
                "shap_importance_all.csv、selected_variables.csv。"
            ),
            citation=_CITE_SHAP,
        ),
        ui.tags.p(
            class_="text-muted",
            style="font-size:0.9rem;",
            *[
                ui.tags.strong("注意："),
                "Beeswarm 圖使用 full-data model 的 SHAP 值（方向參考），"
                "重要性長條圖則來自 OOF SHAP（避免 in-sample 偏誤），兩者估計方式不同。",
            ],
        ),

        # ── 模型來源 ───────────────────────────────────────────
        ui.h4("模型來源"),
        ui.input_radio_buttons(
            "model_source",
            label=None,
            choices={
                "inherit":  "繼承「模型訓練」分頁結果",
                "external": "上傳外部 .pkl 模型",
            },
            selected="inherit",
            inline=True,
        ),
        ui.panel_conditional(
            "input.model_source === 'external'",
            ui.div(
                ui.input_file("model_pkl", "上傳 .pkl 模型檔", accept=[".pkl"]),
                ui.input_file("shap_csv", "上傳對應資料 CSV", accept=[".csv"]),
                ui.input_text("shap_target_column", "目標變項欄名", value="attrition"),
                ui.output_ui("shap_data_summary"),
                class_="alert alert-info",
                style="margin-top: 8px;",
            ),
        ),
        ui.hr(),

        # ── SHAP 參數 ──────────────────────────────────────────
        ui.h4("SHAP 計算參數"),
        ui.layout_columns(
            ui.input_slider("shap_seeds",    "Seeds 數",     min=5,   max=20,   step=1,    value=10),
            ui.input_slider("shap_n_folds",  "OOF Folds",   min=3,   max=10,   step=1,    value=5),
            ui.input_slider("top_n",         "Top-N 顯示",  min=5,   max=50,   step=5,    value=20),
            ui.input_slider("cum_threshold", "累積 SHAP 閾值", min=0.5, max=0.95, step=0.05, value=0.80),
            col_widths=[3, 3, 3, 3],
        ),
        ui.input_task_button("btn_shap", "執行 SHAP"),
        ui.output_ui("shap_progress_bar"),
        ui.output_ui("shap_status"),
        ui.hr(),

        # ── 圖表 ───────────────────────────────────────────────
        ui.h4("圖一：Top-N 變項重要性（OOF SHAP，95% CI）"),
        ui.output_plot("plot_bar", height="500px"),
        ui.br(),

        ui.h4("圖二：Beeswarm（變項效果方向）"),
        ui.output_plot("plot_beeswarm", height="550px"),
        ui.br(),

        ui.h4("圖三：累積 SHAP 比例"),
        ui.output_plot("plot_cum", height="320px"),
        ui.hr(),

        # ── 重要性表格 ─────────────────────────────────────────
        ui.h4("重要性表格"),
        ui.div(
            ui.output_data_frame("shap_table"),
            class_="kit-table-scroll",
        ),
        ui.br(),

        # ── 匯出 ───────────────────────────────────────────────
        ui.h4("匯出"),
        ui.tags.p("逐一下載，或一鍵打包全部 ZIP：", class_="text-muted"),
        ui.layout_columns(
            ui.download_button("dl_shap_all_csv",      "重要性 CSV",  class_="btn-outline-primary"),
            ui.download_button("dl_shap_sel_csv",      "篩選變項 CSV", class_="btn-outline-primary"),
            ui.download_button("dl_shap_bar_png",      "圖一 PNG",    class_="btn-outline-primary"),
            ui.download_button("dl_shap_beeswarm_png", "圖二 PNG",    class_="btn-outline-primary"),
            ui.download_button("dl_shap_cum_png",      "圖三 PNG",    class_="btn-outline-primary"),
            ui.download_button("dl_shap_zip",          "全部 ZIP",    class_="btn-primary"),
            col_widths=[2, 2, 2, 2, 2, 2],
        ),
        download_hint(),
    )


# ── App UI ────────────────────────────────────────────────────────────────────

app_ui = ui.page_fluid(
    # positional children first, then keyword args
    ui.tags.head(ui.tags.style(ui.HTML(
        "@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');"
        "body { font-family: 'Roboto', sans-serif; }"
    ))),
    kit_header(
        title="KIT 流失預測分析工具",
        subtitle="臺灣幼兒發展調查資料庫",
        tagline="FLAML / LightGBM / SHAP",
    ),
    ui.navset_tab(
        _tab_train(),
        _tab_shap(),
        tab_glossary(),
    ),
    theme=make_theme(),
    title="KIT 流失預測分析工具",
)
