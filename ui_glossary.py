"""重要名詞分頁 — KIT 流失預測分析工具."""
from shiny import ui

# ── 每個詞條的資料結構（中英文全名、說明正文、文獻引用）──────────────────────

_TERMS = [
    {
        "id": "flaml",
        "zh": "FLAML",
        "full": "Fast and Lightweight AutoML（快速輕量自動機器學習）",
        "body": (
            "由 Microsoft Research 開發，於 2021 年在 MLSys 發表。"
            "FLAML 整合 <em>BlendSearch</em>（混合式貝葉斯 + 隨機搜索）與 "
            "<em>CFO（Cost-Frugal Optimizer）</em> 兩種自適應搜尋策略，"
            "以最小化試驗成本（而非次數）為核心目標，"
            "在相同或更低的運算預算下顯著超越當時主流的 AutoML 工具（如 auto-sklearn、H2O）。"
            "本工具將其限定於 LightGBM 估計器，"
            "並在時間預算（秒數）內自動搜尋最佳超參數組合。"
        ),
        "cite": (
            "Wang, C., Wu, Q., Weimer, M., &amp; Zhu, E. (2021). "
            "FLAML: A fast and lightweight AutoML library. "
            "<em>Proceedings of Machine Learning and Systems</em>, <em>3</em>, 434–447."
        ),
    },
    {
        "id": "lightgbm",
        "zh": "LightGBM",
        "full": "Light Gradient Boosting Machine（輕量梯度提升機）",
        "body": (
            "Microsoft 研究院 Ke 等人於 2017 年在 NeurIPS 發表。"
            "相較於傳統 GBDT（如 XGBoost），LightGBM 引入兩項關鍵技術："
            "<em>GOSS（Gradient-based One-Side Sampling）</em>——"
            "僅保留梯度較大的樣本以估算資訊增益，大幅降低資料量；"
            "<em>EFB（Exclusive Feature Bundling）</em>——"
            "將互斥特徵捆綁，降低特徵維度。"
            "訓練速度最高可達傳統 GBDT 的 20 倍以上，同時維持近乎相同的精確度，"
            "尤其適合高維、大樣本的表格型資料分類任務。"
        ),
        "cite": (
            "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., &amp; Liu, T. Y. (2017). "
            "LightGBM: A highly efficient gradient boosting decision tree. "
            "<em>Advances in Neural Information Processing Systems</em>, <em>30</em>, 3146–3154."
        ),
    },
    {
        "id": "shap",
        "zh": "SHAP",
        "full": "SHapley Additive exPlanations（夏普利加性解釋）",
        "body": (
            "Lundberg 與 Lee 於 2017 年在 NeurIPS 提出，"
            "奠基於合作賽局理論的 <em>Shapley value</em>（1953）。"
            "其核心思想為：對每個樣本的每個特徵，計算「有該特徵」與「無該特徵」"
            "在所有特徵子集組合下的邊際貢獻期望值，確保 <em>局部準確性</em>、"
            "<em>一致性</em> 與 <em>缺失性</em> 三大公理成立。"
            "本工具使用 <code>shap.TreeExplainer</code>，"
            "針對 LightGBM 樹狀模型提供多項式時間複雜度的精確計算（TreeSHAP）。"
        ),
        "cite": (
            "Lundberg, S. M., &amp; Lee, S.-I. (2017). "
            "A unified approach to interpreting model predictions. "
            "<em>Advances in Neural Information Processing Systems</em>, <em>30</em>, 4765–4774."
            "<br>"
            "Lundberg, S. M., Erion, G., Chen, H., DeYoung, A., Climo, J., Shayda, B., &amp; Lee, S.-I. (2018). "
            "Consistent individualized feature attribution for tree ensembles. "
            "<em>arXiv preprint</em> arXiv:1802.03888."
        ),
    },
    {
        "id": "oof",
        "zh": "OOF 評估",
        "full": "Out-of-Fold（折外）交叉驗證評估",
        "body": (
            "OOF（Out-of-Fold）是 K-Fold 交叉驗證的延伸應用策略："
            "每次以 K−1 折資料訓練模型，對<em>留出折</em>（validation fold）進行預測，"
            "最終將所有折的留出預測值拼接成一組與原始樣本等大的「折外預測」。"
            "此策略確保每筆資料的評估均來自未見過該樣本的模型，"
            "避免 in-sample 偏誤，提供比單次訓練集評估更可靠的泛化誤差估計。"
            "本工具採用 <strong>10 seeds × K folds</strong> 的重複 OOF 設計，"
            "並以各折指標的均值與標準差呈現穩健的模型效能摘要。"
        ),
        "cite": (
            "Bergmeir, C., Hyndman, R. J., &amp; Koo, B. (2018). "
            "A note on the validity of cross-validation for evaluating autoregressive time series prediction. "
            "<em>Computational Statistics &amp; Data Analysis</em>, <em>120</em>, 70–83."
        ),
    },
    {
        "id": "auc",
        "zh": "AUC / AUPRC",
        "full": "Area Under the ROC Curve / Area Under the Precision-Recall Curve（ROC 曲線下面積 / 精確率-召回率曲線下面積）",
        "body": (
            "<strong>AUC-ROC</strong>：以所有分類閾值下「真陽性率 vs. 假陽性率」"
            "所圍的面積衡量模型的整體鑑別力；0.5 為隨機，>0.8 良好，>0.9 優異。"
            "在類別比例嚴重失衡時，AUC 可能過度樂觀，因為大量的真陰性會壓低假陽性率。"
            "<strong>AUPRC</strong>：改以「精確率（Precision）vs. 召回率（Recall）」"
            "為座標，對少數類（流失者）的預測效能更敏感。"
            "本工具以流失率作為 AUPRC 的隨機基線，"
            "並計算模型相對基線的提升倍率（≥ 2× 視為有效）。"
            "建議同時參考兩項指標，全面評估不平衡資料的分類效能。"
        ),
        "cite": (
            "Saito, T., &amp; Rehmsmeier, M. (2015). "
            "The precision-recall plot is more informative than the ROC plot when "
            "evaluating binary classifiers on imbalanced datasets. "
            "<em>PLOS ONE</em>, <em>10</em>(3), e0118432."
            "<br>"
            "Hancock, J. T., &amp; Khoshgoftaar, T. M. (2023). "
            "Evaluating classifier performance with highly imbalanced Big Data. "
            "<em>Journal of Big Data</em>, <em>10</em>, 105."
        ),
    },
    {
        "id": "shap_viz",
        "zh": "SHAP 重要性 / Beeswarm 圖",
        "full": "SHAP Feature Importance Bar Plot / Beeswarm Summary Plot（SHAP 重要性長條圖 / 蜂群效果方向圖）",
        "body": (
            "<strong>重要性長條圖</strong>：以各特徵的 OOF SHAP 絕對值均值（mean |SHAP|）排序，"
            "呈現哪些變項對預測流失風險最有貢獻；本工具另附加 95% CI 誤差棒，"
            "反映跨 seeds/folds 估計的穩定性。"
            "<strong>Beeswarm 圖（蜂群圖）</strong>：以每個樣本一個點呈現 SHAP 值的分布，"
            "橫軸為 SHAP 值（正值 = 增加流失風險），顏色代表特徵原始值高低，"
            "可同時呈現<em>重要性</em>（垂直位置）與<em>效果方向</em>（正/負）。"
            "本工具的 Beeswarm 使用全資料模型的 SHAP 值（方向參考用），"
            "與重要性長條（OOF SHAP）的估計邏輯不同，請勿直接比較數值。"
        ),
        "cite": (
            "Lundberg, S. M., &amp; Lee, S.-I. (2017). "
            "A unified approach to interpreting model predictions. "
            "<em>Advances in Neural Information Processing Systems</em>, <em>30</em>, 4765–4774."
            "<br>"
            "Ponce-Bobadilla, A. V., et al. (2024). "
            "Practical guide to SHAP analysis: Explaining supervised machine learning model "
            "predictions in drug development. "
            "<em>Clinical and Translational Science</em>, <em>17</em>(3), e13706."
        ),
    },
]


def _term_card(term: dict) -> ui.Tag:
    """Render a single glossary card with full name, body, and citation."""
    return ui.div(
        ui.div(
            ui.tags.span(term["zh"], style="font-size:1.15rem; font-weight:600;"),
            ui.tags.span(
                " — ",
                ui.HTML(term["full"]),
                style="color:#555; font-size:0.9rem; margin-left:0.3rem;",
            ),
            style="margin-bottom:0.5rem;",
        ),
        ui.tags.p(
            ui.HTML(term["body"]),
            style="margin-bottom:0.5rem; font-size:0.92rem; line-height:1.65;",
        ),
        ui.div(
            ui.tags.small(
                ui.tags.strong("引用文獻："),
                ui.tags.br(),
                ui.HTML(term["cite"]),
                class_="text-muted",
                style="font-size:0.8rem;",
            ),
            style="border-left:3px solid #2d7535; padding-left:0.75rem; margin-top:0.3rem;",
        ),
        style=(
            "background:#fff; border:1px solid #e0e0e0; border-radius:0.5rem;"
            "padding:1rem 1.25rem; margin-bottom:1rem;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.06);"
        ),
    )


def tab_glossary() -> ui.Tag:
    """「重要名詞」分頁。"""
    cards = [_term_card(t) for t in _TERMS]
    return ui.nav_panel(
        "重要名詞",
        ui.br(),
        ui.div(
            ui.tags.strong("【本分頁目標】"),
            " 提供本工具核心技術名詞的中英文全名、起源文獻、重要概念及實作背景，"
            "供研究團隊參考引用。",
            ui.tags.hr(style="margin:0.5rem 0;"),
            ui.tags.small(
                "各詞條說明約 100–200 字，引用文獻以 APA 格式列出。",
                class_="text-muted",
            ),
            class_="alert alert-success",
            style="margin-bottom:1.25rem;",
        ),
        *cards,
    )
