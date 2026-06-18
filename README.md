# kitMissLGBM：KIT 流失預測分析工具

KIT 風格 [Shiny for Python](https://shiny.posit.co/py/) 應用，整合 **FLAML AutoML**、**LightGBM** 與 **OOF SHAP**，用於 KIT 縱貫資料的後續波次流失（attrition）風險預測與變項重要性分析。

**Repository：** https://github.com/jiewenTsai/kitMissLGBM

---

## 功能

| 分頁 | 功能 |
|------|------|
| **模型訓練** | 上傳 CSV → FLAML 自動調參訓練 LightGBM → OOF 模型評估 → 下載 `.pkl` 與指標 CSV |
| **SHAP 分析** | 繼承訓練結果或上傳外部模型 → OOF SHAP 重要性（含 95% CI）→ 三張圖表 → 下載 CSV / PNG / ZIP |
| **重要名詞** | FLAML、LightGBM、SHAP、OOF、AUC/AUPRC 等核心術語說明與文獻引用 |

---

## 使用：快速開始

### For Windows 

最簡單的方法 (only for windows) 從 GitHub 取得

1. 從右邊 `Releases` 位置下載 zip，到本機上解壓縮。 
2. 進去後直接點擊 `run.bat` 執行即可。

### For Mac/Linux/Win 

都在終端機 terminal 上操作。 

1. 安裝套件

```
git clone https://github.com/jiewenTsai/kitMissLGBM.git
cd kitMissLGBM
pip install -r requirements.txt
```

2. 直接用 python 執行 (請先安裝 python)

```
python app.py
```

<img width="1857" height="990" alt="image" src="https://github.com/user-attachments/assets/2845d36c-1489-414c-8aa7-489ba64d0ea6" />



```powershell
git clone https://github.com/jiewenTsai/kitMissLGBM.git
cd kitMissLGBM
pip install -r requirements.txt
.\run.ps1
```

瀏覽器會自動開啟 http://127.0.0.1:8000 。若尚未安裝 Python，請先看下方 **Windows 安裝 Python**。

---

## 環境需求

- Python 3.10 以上（**建議 3.11 或 3.12**；避免過新的 3.14，部分套件可能尚未完全支援）
- Windows / macOS / Linux

---

## Windows 安裝 Python（首次使用者）

### 步驟 1：安裝 Python

1. 前往 [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. 下載 **Python 3.12**
3. 執行安裝程式時，**務必勾選**「Add python.exe to PATH」
4. 完成安裝後，關閉並重新開啟 PowerShell

### 步驟 2：確認安裝成功

```powershell
python --version
pip --version
```

應顯示類似 `Python 3.12.x`。

### 步驟 3：取得並啟動本工具

```powershell
git clone https://github.com/jiewenTsai/kitMissLGBM.git
cd kitMissLGBM
pip install -r requirements.txt
.\run.ps1
```

若 `git` 指令不存在，可至 [Git for Windows](https://git-scm.com/download/win) 安裝，或改為在 GitHub 網頁按 **Code → Download ZIP** 解壓後進入資料夾執行上述 `pip` 與 `run.ps1`。

### 常見狀況

| 狀況 | 處理方式 |
|------|----------|
| `python` 找不到 | 重新安裝 Python 並勾選 Add to PATH，或改用 `py -3.12` |
| `pip install` 很慢 | 可選：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| PowerShell 無法執行 `run.ps1` | 先執行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，或改雙擊 `run.bat` |
| 瀏覽器沒自動開啟 | 手動開啟 http://127.0.0.1:8000 |

---

## 安裝

```powershell
cd kitMissLGBM
pip install -r requirements.txt
```

首次安裝約需 1–3 分鐘（含 LightGBM、FLAML、SHAP）。

---

## 啟動方式

### 方式一：啟動腳本（推薦，自動開啟瀏覽器）

**PowerShell：**

```powershell
.\run.ps1
```

**命令提示字元 / 雙擊：**

```cmd
run.bat
```

### 方式二：Shiny CLI（需加 `--launch-browser`）

```powershell
python -m shiny run app.py --reload --launch-browser
```

> **注意：** 若只執行 `python -m shiny run app.py --reload`（**沒有** `--launch-browser`），終端機只會顯示網址，**不會**自動開啟瀏覽器。請手動開啟 http://127.0.0.1:8000，或改用上方啟動腳本。

### 方式三：直接執行 app.py

```powershell
python app.py
```

啟動約 1.5 秒後會自動開啟瀏覽器（無熱重載；修改程式碼後需手動重啟）。

---

## GitHub Release 使用方式

每次發佈新版本時，可在 GitHub **Releases** 頁面下載 ZIP 壓縮檔（內含 `run.bat`、`run.ps1`）。

1. 前往 https://github.com/jiewenTsai/kitMissLGBM/releases
2. 下載最新版的 **Source code (zip)**
3. 解壓縮後，在該資料夾開啟 PowerShell
4. 執行：

```powershell
pip install -r requirements.txt
.\run.ps1
```

> Release 僅打包原始碼，**不包含** Python 與套件；首次使用仍需安裝 Python 並執行 `pip install`。

---

## 使用流程

### 步驟 1：模型訓練

1. 開啟 **模型訓練** 分頁
2. 上傳含目標變項 `attrition`（0/1）的 CSV 檔
3. 確認參數：
   - **目標變項欄名**：預設 `attrition`
   - **波次標籤**：用於輸出檔名（如 `M36`）
   - **時間預算**：FLAML 搜尋秒數（預設 300s）
   - **CV folds**：交叉驗證折數（預設 5）
4. 按 **開始訓練**，等待進度條完成（FLAML 搜尋 → OOF 評估）
5. 檢視 **OOF 模型評估摘要** 與 **FLAML 最佳超參數** 表格
6. 下載 **模型 .pkl** 與 **指標 CSV**（可留供日後 SHAP 分析）

### 步驟 2：SHAP 分析

1. 切換至 **SHAP 分析** 分頁
2. 模型來源：
   - **繼承「模型訓練」分頁結果**（同一 session 內訓練完成後直接選此項）
   - **上傳外部 .pkl 模型**（需另上傳對應 CSV）
3. 調整 SHAP 參數（seeds、folds、Top-N、累積閾值）
4. 按 **執行 SHAP**，等待進度條完成
5. 檢視三張圖表與重要性表格
6. 下載個別 CSV / PNG，或 **全部 ZIP**

> **圖表說明：** 圖一（重要性長條圖）使用 OOF SHAP；圖二（Beeswarm）使用 full-data 模型顯示效果方向，兩者估計方式不同。

---

## 輸出檔案

| 檔案 | 說明 |
|------|------|
| `flaml_model_{波次}.pkl` | 訓練完成的 LightGBM 模型 |
| `oof_metrics_summary.csv` | OOF 評估指標（AUC、AUPRC、F1、Recall、Brier） |
| `shap_importance_all.csv` | 全部變項 SHAP 重要性與 95% CI |
| `selected_variables.csv` | 累積 SHAP 閾值內的變項清單 |
| `shap_bar_ci.png` | Top-N 重要性長條圖 |
| `shap_beeswarm.png` | Beeswarm 效果方向圖 |
| `shap_cumulative.png` | 累積 SHAP 比例圖 |
| `shap_all_outputs.zip` | 以上 SHAP 輸出打包 |

若下載無反應，請改用 Chrome 或檢查瀏覽器是否封鎖下載。

---

## 專案結構

```
kitMissLGBM/
├── app.py              # 應用入口
├── ui.py               # 介面定義
├── ui_glossary.py      # 重要名詞分頁
├── server.py           # 互動邏輯與下載
├── ui_helpers.py       # KIT 主題、說明面板
├── run.ps1 / run.bat   # 啟動腳本（自動開瀏覽器）
├── requirements.txt
└── backend/            # 分析後端（無 Shiny 依賴）
    ├── io_data.py      # CSV / 模型讀取
    ├── train.py        # FLAML 訓練、OOF 評估
    ├── shap_analysis.py
    ├── plots.py
    └── export.py
```

---

## 常見問題

**Q：執行 `python -m shiny run app.py --reload` 沒有自動開瀏覽器？**  
A：請加上 `--launch-browser`，或改用 `.\run.ps1` / `run.bat`。

**Q：訓練要多久？**  
A：FLAML 預設時間預算 300 秒，加上 OOF 評估（50 折）約需數分鐘。SHAP 分析（50 折）亦需數分鐘至十餘分鐘，視資料大小而定。

**Q：SHAP 出現特徵數不符錯誤？**  
A：請確認 SHAP 使用的 CSV 與訓練時相同。選「繼承訓練結果」時會自動對齊模型特徵欄位。

**Q：`shiny` 指令找不到？**  
A：改用 `python -m shiny run app.py --reload --launch-browser`。

---

## 分析方法引用

- Wang, C., Wu, Q., Weimer, M., & Zhu, E. (2021). FLAML: A fast and lightweight AutoML library. *Proceedings of Machine Learning and Systems*, *3*, 434–447.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, *30*, 3146–3154.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, *30*, 4765–4774.

---

## 授權

MIT — 詳見 [LICENSE](LICENSE)。
