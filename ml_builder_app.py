# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

st.set_page_config(page_title="AI EA 自作ビルダー (Prototpye)", layout="wide", page_icon="🚀")

# タイトルと説明
st.title("🚀 AI EA 自作モデルビルダー (クオンツ開発ツール)")
st.write(
    "MT5からエクスポートした「1分足ヒストリカルデータ（CSV）」を読み込み、**AI（LightGBM）に学習させたいインジケータ（特徴量）をマウス操作で選択するだけ**で、ご自身専用のAI予測モデル（ONNX）と、それをMT5で動かすためのソースコード（Features.mqh）を自動生成できます。"
)

# ------------------------------------------------------------------
# 1. 特徴量（インジケータ）の日本語定義マッピング
# ------------------------------------------------------------------
FEATURE_MAP = {
    # 5分足ベース特徴量（ラグ処理が自動適用されます）
    "ATR_Ratio": {
        "label": "日足ボラティリティ比率 (価格に対する値幅の大きさ)",
        "category": "日足・時間帯の癖 (Daily & Seasonality)",
        "mql5_handle": "hATR_D1",
        "mql5_init": "hATR_D1 = iATR(smb, PERIOD_D1, 14);",
        "mql5_calc": "        double atr[1];\n        if(CopyBuffer(hATR_D1, 0, shift, 1, atr) < 1) return false;\n        inputs[idx++] = (float)(atr[0] / close_price);"
    },
    "RSI_feat": {
        "label": "RSI (買われすぎ・売られすぎを測るオシレータ)",
        "category": "5分足のインジケータ (M5 Indicators)",
        "mql5_handle": "hRSI_M5",
        "mql5_init": "hRSI_M5 = iRSI(smb, PERIOD_M5, 14, PRICE_CLOSE);",
        "mql5_calc": "        double rsi_buf[1];\n        if(CopyBuffer(hRSI_M5, 0, shift, 1, rsi_buf) < 1) return false;\n        inputs[idx++] = (float)(rsi_buf[0] / 100.0);"
    },
    "Spread_ATR_Ratio": {
        "label": "スプレッドコスト比率 (取引コストがボラティリティに対して適正か)",
        "category": "スプレッド情報 (Spread Info)",
        "mql5_handle": "hATR_D1",
        "mql5_init": "hATR_D1 = iATR(smb, PERIOD_D1, 14);",
        "mql5_calc": "        double atr[1];\n        if(CopyBuffer(hATR_D1, 0, shift, 1, atr) < 1) return false;\n        double spread = iSpread(smb, PERIOD_M5, shift);\n        inputs[idx++] = (float)((spread * point * 10.0) / atr[0]);"
    },
    "Hour_Seasonality": {
        "label": "時間帯周期 (24時間サイクルでの値動きの癖)",
        "category": "日足・時間帯の癖 (Daily & Seasonality)",
        "mql5_handle": "",
        "mql5_init": "",
        "mql5_calc": "        datetime bar_time = iTime(smb, PERIOD_M5, shift);\n        MqlDateTime dt;\n        TimeToStruct(bar_time, dt);\n        inputs[idx++] = (float)MathSin(2.0 * M_PI * dt.hour / 24.0);\n        inputs[idx++] = (float)MathCos(2.0 * M_PI * dt.hour / 24.0);"
    },
    "Day_Seasonality": {
        "label": "曜日周期 (1週間サイクルでの値動きの癖)",
        "category": "日足・時間帯の癖 (Daily & Seasonality)",
        "mql5_handle": "",
        "mql5_init": "",
        "mql5_calc": "        datetime bar_time = iTime(smb, PERIOD_M5, shift);\n        MqlDateTime dt;\n        TimeToStruct(bar_time, dt);\n        int dow = dt.day_of_week;\n        if(dow == 0) dow = 7;\n        inputs[idx++] = (float)MathSin(2.0 * M_PI * dow / 7.0);\n        inputs[idx++] = (float)MathCos(2.0 * M_PI * dow / 7.0);"
    },
    "FracDiff_LogPrice": {
        "label": "分数階微分価格 (価格のトレンド成分を強調抽出しノイズを除去した値)",
        "category": "5分足のインジケータ (M5 Indicators)",
        "mql5_handle": "",
        "mql5_init": "",
        "mql5_calc": "        inputs[idx++] = (float)CalculateFracDiffLogPrice(smb, PERIOD_M5, 0.4, 30, shift);"
    },
    "ADX_feat": {
        "label": "ADX (トレンドの強さ・勢い)",
        "category": "5分足のインジケータ (M5 Indicators)",
        "mql5_handle": "hADX_M5",
        "mql5_init": "hADX_M5 = iADX(smb, PERIOD_M5, 14);",
        "mql5_calc": "        double adx_buf[1];\n        if(CopyBuffer(hADX_M5, 0, shift, 1, adx_buf) < 1) return false;\n        inputs[idx++] = (float)(adx_buf[0] / 100.0);"
    },
    "MACD_Diff_ATR_Ratio": {
        "label": "MACDヒストグラム (トレンドの転換点の予測インジケータ)",
        "category": "5分足のインジケータ (M5 Indicators)",
        "mql5_handle": "hMACD_M5",
        "mql5_init": "hMACD_M5 = iMACD(smb, PERIOD_M5, 12, 26, 9, PRICE_CLOSE);\n    hATR_D1 = iATR(smb, PERIOD_D1, 14);",
        "mql5_calc": "        double macd_main[1], macd_sig[1], atr[1];\n        if(CopyBuffer(hMACD_M5, 0, shift, 1, macd_main) < 1 || CopyBuffer(hMACD_M5, 1, shift, 1, macd_sig) < 1 || CopyBuffer(hATR_D1, 0, shift, 1, atr) < 1) return false;\n        inputs[idx++] = (float)((macd_main[0] - macd_sig[0]) / atr[0]);"
    },
    "BB_Width_ATR_Ratio": {
        "label": "ボリンジャーバンド幅 (スクイーズ/エクスパンションなど相場の収縮・拡張度)",
        "category": "5分足のインジケータ (M5 Indicators)",
        "mql5_handle": "hBB_M5",
        "mql5_init": "hBB_M5 = iBands(smb, PERIOD_M5, 20, 0, 2.0, PRICE_CLOSE);\n    hATR_D1 = iATR(smb, PERIOD_D1, 14);",
        "mql5_calc": "        double bb_up[1], bb_down[1], atr[1];\n        if(CopyBuffer(hBB_M5, 1, shift, 1, bb_up) < 1 || CopyBuffer(hBB_M5, 2, shift, 1, bb_down) < 1 || CopyBuffer(hATR_D1, 0, shift, 1, atr) < 1) return false;\n        inputs[idx++] = (float)((bb_up[0] - bb_down[0]) / atr[0]);"
    },
    "EMA_Diff_ATR_Ratio": {
        "label": "EMA 200 乖離率 (長期トレンド方向からの価格の離れ具合)",
        "category": "5分足のインジケータ (M5 Indicators)",
        "mql5_handle": "hEMA_M5",
        "mql5_init": "hEMA_M5 = iMA(smb, PERIOD_M5, 200, 0, MODE_EMA, PRICE_CLOSE);\n    hATR_D1 = iATR(smb, PERIOD_D1, 14);",
        "mql5_calc": "        double ema_val[1], atr[1];\n        if(CopyBuffer(hEMA_M5, 0, shift, 1, ema_val) < 1 || CopyBuffer(hATR_D1, 0, shift, 1, atr) < 1) return false;\n        inputs[idx++] = (float)((close_price - ema_val[0]) / atr[0]);"
    },
    "Hour_Activity_feat": {
        "label": "市場活動度 (時間帯別の平均取引量の大きさ)",
        "category": "日足・時間帯の癖 (Daily & Seasonality)",
        "mql5_handle": "",
        "mql5_init": "",
        "mql5_calc": "        datetime bar_time = iTime(smb, PERIOD_M5, shift);\n        MqlDateTime dt;\n        TimeToStruct(bar_time, dt);\n        int hour_activity_map[24] = {7, 9, 8, 5, 2, 3, 5, 6, 6, 6, 6, 6, 8, 10, 10, 9, 8, 7, 6, 6, 4, 3, 5, 7};\n        inputs[idx++] = (float)(hour_activity_map[dt.hour] / 10.0);"
    },
    "Momentum_5": {
        "label": "モメンタム 5本 (直近のローソク足の勢い)",
        "category": "5分足のインジケータ (M5 Indicators)",
        "mql5_handle": "hATR_M5",
        "mql5_init": "hATR_M5 = iATR(smb, PERIOD_M5, 14);",
        "mql5_calc": "        double atr_tf[1];\n        if(CopyBuffer(hATR_M5, 0, shift, 1, atr_tf) < 1) return false;\n        double prev_close_5 = iClose(smb, PERIOD_M5, shift + 5);\n        inputs[idx++] = (float)((close_price - prev_close_5) / (atr_tf[0] + 1e-10));"
    },
    "Momentum_15": {
        "label": "モメンタム 15本 (中期ローソク足の勢い)",
        "category": "5分足のインジケータ (M5 Indicators)",
        "mql5_handle": "hATR_M5",
        "mql5_init": "hATR_M5 = iATR(smb, PERIOD_M5, 14);",
        "mql5_calc": "        double atr_tf[1];\n        if(CopyBuffer(hATR_M5, 0, shift, 1, atr_tf) < 1) return false;\n        double prev_close_15 = iClose(smb, PERIOD_M5, shift + 15);\n        inputs[idx++] = (float)((close_price - prev_close_15) / (atr_tf[0] + 1e-10));"
    },
    
    # 上位足インジケータ (マルチタイムフレーム処理 - shift 1 のみ)
    "RSI_60m": {
        "label": "1時間足 RSI (長期オシレータ方向)",
        "category": "上位足インジケータ (1時間足 / H1)",
        "mql5_handle": "hRSI_H1",
        "mql5_init": "hRSI_H1 = iRSI(smb, PERIOD_H1, 14, PRICE_CLOSE);",
        "mql5_calc_mtf": "    double rsi_h1[1];\n    if(CopyBuffer(hRSI_H1, 0, 1, 1, rsi_h1) < 1) return false;\n    inputs[idx++] = (float)(rsi_h1[0] / 100.0);"
    },
    "ADX_60m": {
        "label": "1時間足 ADX (長期トレンドの強度)",
        "category": "上位足インジケータ (1時間足 / H1)",
        "mql5_handle": "hADX_H1",
        "mql5_init": "hADX_H1 = iADX(smb, PERIOD_H1, 14);",
        "mql5_calc_mtf": "    double adx_h1[1];\n    if(CopyBuffer(hADX_H1, 0, 1, 1, adx_h1) < 1) return false;\n    inputs[idx++] = (float)(adx_h1[0] / 100.0);"
    },
    "EMA_Diff_ATR_Ratio_60m": {
        "label": "1時間足 EMA200 乖離率 (長期的な移動平均との乖離)",
        "category": "上位足インジケータ (1時間足 / H1)",
        "mql5_handle": "hEMA_H1",
        "mql5_init": "hEMA_H1 = iMA(smb, PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE);\n    hATR_H1 = iATR(smb, PERIOD_H1, 14);",
        "mql5_calc_mtf": "    double ema_h1[1], atr_h1[1];\n    if(CopyBuffer(hEMA_H1, 0, 1, 1, ema_h1) < 1 || CopyBuffer(hATR_H1, 0, 1, 1, atr_h1) < 1) return false;\n    double close_h1 = iClose(smb, PERIOD_H1, 1);\n    inputs[idx++] = (float)((close_h1 - ema_h1[0]) / atr_h1[0]);"
    },
    "RSI_240m": {
        "label": "4時間足 RSI (超長期オシレータ方向)",
        "category": "上位足インジケータ (4時間足 / H4)",
        "mql5_handle": "hRSI_H4",
        "mql5_init": "hRSI_H4 = iRSI(smb, PERIOD_H4, 14, PRICE_CLOSE);",
        "mql5_calc_mtf": "    double rsi_h4[1];\n    if(CopyBuffer(hRSI_H4, 0, 1, 1, rsi_h4) < 1) return false;\n    inputs[idx++] = (float)(rsi_h4[0] / 100.0);"
    },
    "ADX_240m": {
        "label": "4時間足 ADX (超長期トレンドの強度)",
        "category": "上位足インジケータ (4時間足 / H4)",
        "mql5_handle": "hADX_H4",
        "mql5_init": "hADX_H4 = iADX(smb, PERIOD_H4, 14);",
        "mql5_calc_mtf": "    double adx_h4[1];\n    if(CopyBuffer(hADX_H4, 0, 1, 1, adx_h4) < 1) return false;\n    inputs[idx++] = (float)(adx_h4[0] / 100.0);"
    },
    "EMA_Diff_ATR_Ratio_240m": {
        "label": "4時間足 EMA200 乖離率 (超長期移動平均との乖離)",
        "category": "上位足インジケータ (4時間足 / H4)",
        "mql5_handle": "hEMA_H4",
        "mql5_init": "hEMA_H4 = iMA(smb, PERIOD_H4, 200, 0, MODE_EMA, PRICE_CLOSE);\n    hATR_H4 = iATR(smb, PERIOD_H4, 14);",
        "mql5_calc_mtf": "    double ema_h4[1], atr_h4[1];\n    if(CopyBuffer(hEMA_H4, 0, 1, 1, ema_h4) < 1 || CopyBuffer(hATR_H4, 0, 1, 1, atr_h4) < 1) return false;\n    double close_h4 = iClose(smb, PERIOD_H4, 1);\n    inputs[idx++] = (float)((close_h4 - ema_h4[0]) / atr_h4[0]);"
    }
}

# ------------------------------------------------------------------
# 1.5 カスタムインジケータ（custom_features.py）の自動ロード
# ------------------------------------------------------------------
try:
    if os.path.exists("custom_features.py"):
        import custom_features
        import importlib
        importlib.reload(custom_features)
        if hasattr(custom_features, "CUSTOM_INDICATORS"):
            for key, val in custom_features.CUSTOM_INDICATORS.items():
                FEATURE_MAP[key] = val
            st.sidebar.success(f"✅ 独自カスタム特徴量 ({len(custom_features.CUSTOM_INDICATORS)}個) のロードに成功！")
except Exception as e:
    st.sidebar.warning(f"⚠️ カスタム特徴量の読み込みエラー: {e}")

# ------------------------------------------------------------------
# 2. サイドバー設定 (初心者向け説明付き)
# ------------------------------------------------------------------
st.sidebar.header("⚙️ システム基本設定")

# CSVデータ自動検出
csv_files = glob.glob("*_M1*.csv")
if not csv_files:
    st.sidebar.error("CSVデータが見つかりません。MT5から出力した『BTCUSD_M1.csv』などのファイルをフォルダ内に置いてください。")
    selected_csv = None
else:
    selected_csv = st.sidebar.selectbox("📂 ヒストリカルデータ (M1 CSV) の選択", csv_files, help="MT5からエクスポートした学習用データを選択します。")

st.sidebar.subheader("🎯 利確・損切りの目安時間 (AIターゲット)")
barrier_mult = st.sidebar.slider("AIが狙う目標値幅の広さ (ボラティリティ乗数)", 0.10, 0.50, 0.20, 0.05, help="インジケータのATRに対する比率。数値を大きくすると長期保有（大損小利・大利小損）になり、小さくするとスキャルピング向きになります。通常0.20が標準です。")
horizon_m5 = st.sidebar.number_input("5分足での最大保有時間 (バー本数)", 30, 240, 120, 10, help="この本数の間に目標値幅に達しない場合はレンジ決済と判定。120本で10時間に対応。")
horizon_m15 = st.sidebar.number_input("15分足での最大保有時間 (バー本数)", 120, 720, 360, 20, help="15分足時の判定用。360本で90時間に対応。")

st.sidebar.subheader("🧠 AIの学習細かさ調整 (上級者用)")
learning_rate = st.sidebar.slider("学習スピード (Learning Rate)", 0.01, 0.10, 0.03, 0.01, help="AIがパターンをどれだけ細かく学ぶか。小さいほど高精度ですが時間がかかります。デフォルトは0.03です。")
max_depth = st.sidebar.slider("AI意思決定ツリーの深さ (Max Depth)", 3, 10, 6, 1, help="ツリーの最大深さ。大きいほど複雑な条件を覚えますが、過学習（過去データだけに強くなる現象）のリスクが高まります。")
num_leaves = st.sidebar.slider("ツリーの分岐数 (Num Leaves)", 15, 127, 31, 2, help="決定ツリーの分岐点。31が標準です。")
n_estimators = st.sidebar.number_input("ツリー構築数 (Estimators)", 50, 1000, 300, 50, help="決定ツリーの合計本数。多いほど予測が安定します。")

# ------------------------------------------------------------------
# 3. メイン画面 特徴量チェックボックスグリッド
# ------------------------------------------------------------------
st.header("🛠️ ステップ1: AIに読み込ませるインジケータの選択")
st.write("AIに値動きの予測判断材料として使わせたいインジケータにチェックを入れてください。")

categories = [
    "5分足のインジケータ (M5 Indicators)",
    "日足・時間帯の癖 (Daily & Seasonality)",
    "スプレッド情報 (Spread Info)",
    "上位足インジケータ (1時間足 / H1)",
    "上位足インジケータ (4時間足 / H4)"
]

# 追加されたカスタムカテゴリを動的登録
for f_name, f_info in FEATURE_MAP.items():
    if f_info["category"] not in categories:
        categories.append(f_info["category"])

selected_features = []

cols = st.columns(len(categories))
for col_idx, category in enumerate(categories):
    with cols[col_idx]:
        st.subheader(category)
        for f_name, f_info in FEATURE_MAP.items():
            if f_info["category"] == category:
                checked = st.checkbox(f_info["label"], value=True, key=f_name)
                if checked:
                    selected_features.append(f_name)

# ------------------------------------------------------------------
# 4. 次元数要約と説明
# ------------------------------------------------------------------
base_m5_selected = [f for f in selected_features if not f.endswith("60m") and not f.endswith("240m") and FEATURE_MAP[f]["category"] not in ["上位足インジケータ (1時間足 / H1)", "上位足インジケータ (4時間足 / H4)"]]
mtf_selected = [f for f in selected_features if f not in base_m5_selected]

# 5分足ベース特徴量は自動で4個のラグ（0, 1, 3, 7本前）を生成して学習します
dim_base = len(base_m5_selected) * 4
dim_mtf = len(mtf_selected)
total_dim = dim_base + dim_mtf

st.info(
    f"📊 **【特徴量の次元数（AIモデルの入力データ）】**\n"
    f"- 5分足インジケータ: {len(base_m5_selected)}個 × 4つの過去履歴 (現在、1本前、3本前、7本前) = {dim_base}次元\n"
    f"- 上位足 (1時間/4時間足) インジケータ: {len(mtf_selected)}次元\n"
    f"- **AIの予測条件データの次元数 (ONNXモデル形状): [None, {total_dim}]**"
)

# ------------------------------------------------------------------
# 5. モデル学習パイプライン
# ------------------------------------------------------------------
st.header("🤖 ステップ2: AIのトレーニング開始とコード書き出し")

if st.button("🚀 AIの学習処理を開始する", disabled=(selected_csv is None or total_dim == 0)):
    symbol = os.path.basename(selected_csv).split('_M1')[0]
    
    with st.spinner(f"データセットを計算し、{symbol} の学習モデルを構築中..."):
        
        # Indicator calculation functions
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))

        def calculate_atr(df, period=14):
            high, low, prev_close = df['High'], df['Low'], df['Close'].shift(1)
            tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            return tr.rolling(window=period).mean()

        def calculate_macd(series, fast=12, slow=26, signal=9):
            macd_line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
            return macd_line - macd_line.ewm(span=signal, adjust=False).mean()

        def calculate_bb_width(series, period=20, num_std=2.0):
            sma = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            return (sma + std * num_std) - (sma - std * num_std)

        def calculate_ema(series, period=200):
            return series.ewm(span=period, adjust=False).mean()

        def calculate_adx(df, period=14):
            high, low, close = df['High'], df['Low'], df['Close']
            tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            alpha = 1.0 / period
            atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean() / (atr_s + 1e-10)
            minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean() / (atr_s + 1e-10)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
            return dx.ewm(alpha=alpha, adjust=False).mean()

        def apply_frac_diff(series_values, d=0.4, window=30):
            w = [1.0]
            for k in range(1, window):
                w.append(-w[-1] / k * (d - k + 1))
            w_rev = np.array(w)[::-1]
            res = np.convolve(series_values, w_rev, mode='valid')
            return np.concatenate([np.full(window - 1, np.nan), res])

        def calculate_triple_barrier(close, high, low, atr, H, mult):
            n = len(close)
            labels = np.zeros(n, dtype=np.int32)
            for i in range(n - H):
                if atr[i] <= 0 or np.isnan(atr[i]):
                    continue
                u = close[i] + atr[i] * mult
                l = close[i] - atr[i] * mult
                sub_high = high[i+1 : i+H+1]
                sub_low  = low[i+1 : i+H+1]
                up_touches = np.where(sub_high >= u)[0]
                down_touches = np.where(sub_low <= l)[0]
                first_up = up_touches[0] if len(up_touches) > 0 else H
                first_down = down_touches[0] if len(down_touches) > 0 else H
                if first_up < first_down:
                    labels[i] = 1 # UP
                elif first_down < first_up:
                    labels[i] = 2 # DOWN
                else:
                    labels[i] = 0 # Range/Flat
            return labels

        # Read CSV data
        df = pd.read_csv(selected_csv, sep='\t')
        df.columns = [c.strip().replace('<', '').replace('>', '') for c in df.columns]
        rename_map = {'DATE':'Date','TIME':'Time','OPEN':'Open','HIGH':'High','LOW':'Low','CLOSE':'Close','TICKVOL':'Volume','SPREAD':'Spread'}
        df = df.rename(columns=rename_map)
        
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        else:
            df['DateTime'] = pd.to_datetime(df.iloc[:, 0])
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        # Digits/Spread multi-plier detection
        sample = df['Close'].dropna().head(1000).astype(str)
        digits = 0
        for val in sample:
            if '.' in val:
                decimals = len(val.split('.')[1])
                if decimals > digits: digits = decimals
        spread_mult = 10 ** (1 - digits)

        # 1. Daily ATR
        df_d1 = df.set_index('DateTime').resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna().reset_index()
        df_d1['ATR_D1'] = calculate_atr(df_d1, 14).shift(1)
        df_d1 = df_d1[['DateTime', 'ATR_D1']].dropna()

        # 2. Labeling
        df_m1_temp = pd.merge_asof(df.sort_values('DateTime'), df_d1.sort_values('DateTime'), on='DateTime', direction='backward')
        y_m5_all = calculate_triple_barrier(df_m1_temp['Close'].values, df_m1_temp['High'].values, df_m1_temp['Low'].values, df_m1_temp['ATR_D1'].values, H=horizon_m5, mult=barrier_mult)
        y_m15_all = calculate_triple_barrier(df_m1_temp['Close'].values, df_m1_temp['High'].values, df_m1_temp['Low'].values, df_m1_temp['ATR_D1'].values, H=horizon_m15, mult=barrier_mult)

        # 3. Higher timeframe features (H1/H4)
        df_indexed = df.set_index('DateTime')
        def compute_mtf_features(tf_str, s_label):
            tf = df_indexed.resample(tf_str).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna().reset_index()
            tf[f'RSI_{s_label}'] = calculate_rsi(tf['Close'], 14) / 100.0
            tf[f'ADX_{s_label}'] = calculate_adx(tf, 14) / 100.0
            tf[f'EMA_Diff_ATR_Ratio_{s_label}'] = (tf['Close'] - calculate_ema(tf['Close'], 200)) / (calculate_atr(tf, 14) + 1e-10)
            cols = [f'RSI_{s_label}', f'ADX_{s_label}', f'EMA_Diff_ATR_Ratio_{s_label}']
            tf[cols] = tf[cols].shift(1)
            return tf[['DateTime'] + cols]
        
        df_h1_feats = compute_mtf_features('60min', '60m')
        df_h4_feats = compute_mtf_features('240min', '240m')

        # 4. Feature Construction
        def compute_dataset(df_m1, label_array, tf_str, horizon):
            df_tf = df_m1.set_index('DateTime').resample(tf_str).agg({'Open':'first','High':'max','Low':'min','Close':'last','Spread':'mean'}).dropna().reset_index()
            df_tf = pd.merge_asof(df_tf.sort_values('DateTime'), df_d1.sort_values('DateTime'), on='DateTime', direction='backward')
            
            # Base indicator calculations (Standard)
            df_tf['ATR_Ratio'] = df_tf['ATR_D1'] / df_tf['Close']
            df_tf['RSI_feat'] = calculate_rsi(df_tf['Close'], 14) / 100.0
            df_tf['Spread_ATR_Ratio'] = (df_tf['Spread'] * spread_mult) / df_tf['ATR_D1']
            
            df_tf['Hour'] = df_tf['DateTime'].dt.hour
            df_tf['Hour_Sin_feat'] = np.sin(2 * np.pi * df_tf['Hour'] / 24.0)
            df_tf['Hour_Cos_feat'] = np.cos(2 * np.pi * df_tf['Hour'] / 24.0)
            
            df_tf['DayOfWeek'] = df_tf['DateTime'].dt.dayofweek
            df_tf['Day_Sin_feat'] = np.sin(2 * np.pi * df_tf['DayOfWeek'] / 7.0)
            df_tf['Day_Cos_feat'] = np.cos(2 * np.pi * df_tf['DayOfWeek'] / 7.0)
            
            df_tf['FracDiff_LogPrice'] = apply_frac_diff(np.log(df_tf['Close'].values))
            df_tf['ADX_feat'] = calculate_adx(df_tf, 14) / 100.0
            df_tf['MACD_Diff_ATR_Ratio'] = calculate_macd(df_tf['Close']) / df_tf['ATR_D1']
            df_tf['BB_Width_ATR_Ratio'] = calculate_bb_width(df_tf['Close']) / df_tf['ATR_D1']
            df_tf['EMA_Diff_ATR_Ratio'] = (df_tf['Close'] - calculate_ema(df_tf['Close'], 200)) / df_tf['ATR_D1']
            
            hour_activity_map = {0:0.7, 1:0.9, 2:0.8, 3:0.5, 4:0.2, 5:0.3, 6:0.5, 7:0.6, 8:0.6, 9:0.6, 10:0.6, 11:0.6, 12:0.8, 13:1.0, 14:1.0, 15:0.9, 16:0.8, 17:0.7, 18:0.6, 19:0.6, 20:0.4, 21:0.3, 22:0.5, 23:0.7}
            df_tf['Hour_Activity_feat'] = df_tf['Hour'].map(hour_activity_map)
            
            atr_tf = calculate_atr(df_tf, 14) + 1e-10
            df_tf['Momentum_5'] = (df_tf['Close'] - df_tf['Close'].shift(5)) / atr_tf
            df_tf['Momentum_15'] = (df_tf['Close'] - df_tf['Close'].shift(15)) / atr_tf
            
            # Map Custom indicators dynamically
            for f in base_m5_selected:
                if f in FEATURE_MAP and "py_calc" in FEATURE_MAP[f]:
                    df_tf[f] = FEATURE_MAP[f]["py_calc"](df_tf)
            
            # Map Python columns
            python_cols_map = {
                "ATR_Ratio": "ATR_Ratio", "RSI_feat": "RSI_feat", "Spread_ATR_Ratio": "Spread_ATR_Ratio",
                "Hour_Seasonality": ["Hour_Sin_feat", "Hour_Cos_feat"],
                "Day_Seasonality": ["Day_Sin_feat", "Day_Cos_feat"],
                "FracDiff_LogPrice": "FracDiff_LogPrice", "ADX_feat": "ADX_feat",
                "MACD_Diff_ATR_Ratio": "MACD_Diff_ATR_Ratio", "BB_Width_ATR_Ratio": "BB_Width_ATR_Ratio",
                "EMA_Diff_ATR_Ratio": "EMA_Diff_ATR_Ratio", "Hour_Activity_feat": "Hour_Activity_feat",
                "Momentum_5": "Momentum_5", "Momentum_15": "Momentum_15"
            }
            
            # Get only selected base features
            base_cols_to_shift = []
            for f in base_m5_selected:
                if f in python_cols_map:
                    mapped = python_cols_map[f]
                    if isinstance(mapped, list):
                        base_cols_to_shift.extend(mapped)
                    else:
                        base_cols_to_shift.append(mapped)
                else:
                    # Custom feature key is its column name
                    base_cols_to_shift.append(f)
            
            # Shift base features to simulate closed historical bar values only
            df_tf[base_cols_to_shift] = df_tf[base_cols_to_shift].shift(1)
            
            # Create lags (0, 1, 3, 7 shifts)
            shifts = [0, 1, 3, 7]
            lag_cols = []
            for s_val in shifts:
                for col in base_cols_to_shift:
                    col_name = f"{col}_lag_{s_val}"
                    df_tf[col_name] = df_tf[col].shift(s_val)
                    lag_cols.append(col_name)
            
            # Merge higher timeframe features
            df_tf = pd.merge_asof(df_tf.sort_values('DateTime'), df_h1_feats.sort_values('DateTime'), on='DateTime', direction='backward')
            df_tf = pd.merge_asof(df_tf.sort_values('DateTime'), df_h4_feats.sort_values('DateTime'), on='DateTime', direction='backward')
            
            # Filter selected MTF columns
            mtf_cols = []
            for f in mtf_selected:
                mtf_cols.append(f)
                
            all_feat_cols = lag_cols + mtf_cols
            df_tf_clean = df_tf.dropna(subset=all_feat_cols).reset_index(drop=True)
            
            # Merge M1 labels
            df_m1_labels = df_m1[['DateTime', 'Label']].copy()
            df_tf_clean = pd.merge_asof(df_tf_clean.sort_values('DateTime'), df_m1_labels.sort_values('DateTime'), on='DateTime', direction='backward')
            
            X = df_tf_clean[all_feat_cols].values.astype(np.float32)
            y = df_tf_clean['Label'].values.astype(np.int64)
            return X, y, all_feat_cols

        df_m5_m1 = df.copy(); df_m5_m1['Label'] = y_m5_all
        X_m5, y_m5, feat_names = compute_dataset(df_m5_m1, y_m5_all, '5min', horizon_m5)
        
        df_m15_m1 = df.copy(); df_m15_m1['Label'] = y_m15_all
        X_m15, y_m15, _ = compute_dataset(df_m15_m1, y_m15_all, '15min', horizon_m15)
        
        X = np.vstack([X_m5, X_m15]).astype(np.float32)
        y = np.concatenate([y_m5, y_m15]).astype(np.int64)

        # Train/Test Split (70% Train, 15% Val, 15% Test)
        n_samples = len(X)
        train_size = int(n_samples * 0.70)
        val_size = int(n_samples * 0.15)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

        # Train LightGBM model
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            n_jobs=1
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(15, verbose=False)])
        best_iter = model.best_iteration_

        # Retrain on full dataset
        final_model = lgb.LGBMClassifier(
            n_estimators=best_iter if best_iter > 0 else 100,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            n_jobs=1
        )
        final_model.fit(X, y)

        # Evaluate model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Range', 'UP', 'DOWN'], output_dict=True)

        # --------------------------------------------------------------
        # EXPORT 1: ONNX Model
        # --------------------------------------------------------------
        onnx_filename = f"ml_model_{symbol}_custom.onnx"
        initial_types = [('input', FloatTensorType([None, total_dim]))]
        onnx_model = onnxmltools.convert_lightgbm(final_model, initial_types=initial_types, target_opset=12, zipmap=False)
        onnxmltools.utils.save_model(onnx_model, onnx_filename)

        # --------------------------------------------------------------
        # EXPORT 2: Dynamic MQL5 Header File (*.mqh)
        # --------------------------------------------------------------
        mqh_filename = f"Features_{symbol}_custom.mqh"
        
        # Build Handles, Initializations, Releases, and Validations
        handles_set = set()
        handle_decls = []
        handle_inits = []
        handle_releases = []
        handle_validations = []
        
        # Scan chosen features for handles
        all_chosen = selected_features
        for f in all_chosen:
            f_info = FEATURE_MAP[f]
            if "mql5_handle_decls" in f_info:
                # Custom feature definitions drop-in
                handle_decls.append(f_info["mql5_handle_decls"])
                handle_inits.append(f_info["mql5_init"])
                handle_releases.append(f_info["mql5_release"])
                handle_validations.append(f_info["mql5_validation"])
                # Mark custom handles in handles_set to prevent doubles
                if "mql5_handle" in f_info and f_info["mql5_handle"]:
                    handles = [h.strip() for h in f_info["mql5_handle"].split(",") if h.strip()]
                    for h in handles: handles_set.add(h)
            else:
                # Standard feature handles
                h_name = f_info.get("mql5_handle")
                if h_name and h_name not in handles_set:
                    handles_set.add(h_name)
                    handle_decls.append(f"int {h_name};")
                    handle_inits.append(f"    {f_info['mql5_init']}")
                    handle_releases.append(f"    IndicatorRelease({h_name});")
                    handle_validations.append(f"    if({h_name} == INVALID_HANDLE) return false;")
        
        # EMA_Diff_ATR_Ratio_60m and EMA_Diff_ATR_Ratio_240m require special handles in MQL5
        if "EMA_Diff_ATR_Ratio_60m" in all_chosen:
            if "hATR_H1" not in handles_set:
                handle_decls.append("int hATR_H1;")
                handle_inits.append("    hATR_H1 = iATR(smb, PERIOD_H1, 14);")
                handle_releases.append("    IndicatorRelease(hATR_H1);")
                handle_validations.append("    if(hATR_H1 == INVALID_HANDLE) return false;")
                handles_set.add("hATR_H1")
        if "EMA_Diff_ATR_Ratio_240m" in all_chosen:
            if "hATR_H4" not in handles_set:
                handle_decls.append("int hATR_H4;")
                handle_inits.append("    hATR_H4 = iATR(smb, PERIOD_H4, 14);")
                handle_releases.append("    IndicatorRelease(hATR_H4);")
                handle_validations.append("    if(hATR_H4 == INVALID_HANDLE) return false;")
                handles_set.add("hATR_H4")
        
        # Multi_Symbol EA also needs D1 ATR for spread scaling and volatility calculations
        if "hATR_D1" not in handles_set:
            handle_decls.append("int hATR_D1;")
            handle_inits.append("    hATR_D1 = iATR(smb, PERIOD_D1, 14);")
            handle_releases.append("    IndicatorRelease(hATR_D1);")
            handle_validations.append("    if(hATR_D1 == INVALID_HANDLE) return false;")
            handles_set.add("hATR_D1")

        handle_declarations_str = "\n".join(handle_decls)
        handle_initializations_str = "\n".join(handle_inits)
        handle_releases_str = "\n".join(handle_releases)
        handle_validation_str = "\n".join(handle_validations)

        # Base Feature calculations code block
        base_calc_lines = []
        for f in base_m5_selected:
            base_calc_lines.append(f"        // --- {f} ---\n{FEATURE_MAP[f]['mql5_calc']}\n")
        base_calc_str = "\n".join(base_calc_lines)

        # MTF Feature calculations code block
        mtf_calc_lines = []
        for f in mtf_selected:
            if "mql5_calc_mtf" in FEATURE_MAP[f]:
                mtf_calc_lines.append(f"    // --- {f} ---\n{FEATURE_MAP[f]['mql5_calc_mtf']}\n")
            else:
                # If custom feature is marked in MTF category
                mtf_calc_lines.append(f"    // --- {f} ---\n{FEATURE_MAP[f]['mql5_calc']}\n")
        mtf_calc_str = "\n".join(mtf_calc_lines)

        # Combine into complete MQH template
        mqh_content = f"""//+------------------------------------------------------------------+
//|                                           {mqh_filename} |
//|        Generated automatically by AI Trader Custom ML Builder    |
//+------------------------------------------------------------------+
#property strict

#define FEATURE_COUNT {total_dim}

//--- Indicator Handles
{handle_declarations_str}

//+------------------------------------------------------------------+
//| Initialize handles for the selected features                     |
//+------------------------------------------------------------------+
void InitFeatureHandles(string smb)
{{
{handle_initializations_str}
}}

//+------------------------------------------------------------------+
//| Release handles                                                  |
//+------------------------------------------------------------------+
void ReleaseFeatureHandles()
{{
{handle_releases_str}
}}

//+------------------------------------------------------------------+
//| Helper: Calculate Fractional Difference of Log Price            |
//+------------------------------------------------------------------+
double CalculateFracDiffLogPrice(string symbol, ENUM_TIMEFRAMES tf, double d, int window, int start_shift)
{{
    double close_prices[];
    ArraySetAsSeries(close_prices, true);
    if(CopyClose(symbol, tf, start_shift, window, close_prices) < window) {{
        return 0.0;
    }}
    double w[];
    ArrayResize(w, window);
    w[0] = 1.0;
    for(int k=1; k<window; k++) {{
        w[k] = -w[k-1] / k * (d - k + 1);
    }}
    double frac_diff = 0.0;
    for(int k=0; k<window; k++) {{
        if(close_prices[k] <= 0) return 0.0;
        frac_diff += w[k] * MathLog(close_prices[k]);
    }}
    return frac_diff;
}}

//+------------------------------------------------------------------+
//| Construct the input feature array for the ONNX model             |
//+------------------------------------------------------------------+
bool ConstructCustomInputs(string smb, float &inputs[], double point)
{{
    int idx = 0;
    int shifts[4] = {{1, 2, 4, 8}};
    
    // Validate handles
{handle_validation_str}

    for(int lag = 0; lag < 4; lag++)
    {{
        int shift = shifts[lag];
        double close_price = iClose(smb, PERIOD_M5, shift);
        if(close_price <= 0) return false;
        
{base_calc_str}
    }}
    
    // Multi-timeframe features (shift 1)
{mtf_calc_str}

    return (idx == FEATURE_COUNT);
}}
"""
        with open(mqh_filename, "w", encoding="utf-8") as f:
            f.write(mqh_content)

        # --------------------------------------------------------------
        # UI Metrics Display (日本語化)
        # --------------------------------------------------------------
        st.success("🎉 AIモデルの学習、およびファイル出力が完了しました！")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("検証データ予測正解率 (Accuracy)", f"{acc*100:.2f}%")
        with col2:
            st.metric("買いシグナル(UP) の精度 (Precision)", f"{report['UP']['precision']*100:.2f}%")
        with col3:
            st.metric("売りシグナル(DOWN) の精度 (Precision)", f"{report['DOWN']['precision']*100:.2f}%")
        
        # Classification report in table
        st.subheader("📊 詳細評価レポート")
        df_report = pd.DataFrame(report).transpose().iloc[:3]
        df_report.index = ["レンジ判定 (Range)", "上昇予測 (UP)", "下落予測 (DOWN)"]
        df_report.columns = ["適合率 (Precision)", "再現率 (Recall)", "F1スコア (F1-score)", "テストデータ数 (Support)"]
        st.dataframe(df_report.style.format("{:.4f}"))

        # Feature Importance Plot
        st.subheader("🔥 特徴量重要度 (どのインジケータがAIの予測に効いているか)")
        importance = model.feature_importances_
        df_imp = pd.DataFrame({"インジケータ": feat_names, "貢献度": importance}).sort_values("貢献度", ascending=False)
        st.bar_chart(df_imp.set_index("インジケータ"))

        # Output Files Display
        st.subheader("💾 生成されたファイル")
        st.write(f"1. **ONNXモデル (AI本体)**: `{os.path.abspath(onnx_filename)}`")
        st.write(f"2. **MQL5ヘッダー (計算式コード)**: `{os.path.abspath(mqh_filename)}`")
        
        # Provide code preview
        with st.expander("📝 生成された MQL5 ソースコードのプレビュー (Features.mqh)"):
            st.code(mqh_content, language="mql5")
