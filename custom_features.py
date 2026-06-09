# -*- coding: utf-8 -*-
# ==================================================================
# custom_features.py
# ==================================================================
# ユーザーや開発者が後から「独自のインジケータ（特徴量）」を追加するためのファイルです。
# ここに定義を追加するだけで、アプリの画面に自動的にチェックボックスが現れ、
# 機械学習のデータセットおよび出力されるMQL5コード（Features.mqh）に自動反映されます。

CUSTOM_INDICATORS = {
    # --------------------------------------------------------------
    # 例: 5分足のEMAのゴールデンクロス/デッドクロス（乖離率ベース）
    # --------------------------------------------------------------
    "EMA_Cross_feat": {
        "label": "EMA 5 & 20 Cross Distance (カスタム指標例)",
        "category": "Custom Indicators",
        
        # 1. Python (pandas) での特徴量計算ロジック
        # ※ 引数dfは、Open/High/Low/Closeなどの列を持つ1分足またはResample済みのDataFrame
        "py_calc": lambda df: (df['Close'].ewm(span=5, adjust=False).mean() - df['Close'].ewm(span=20, adjust=False).mean()) / (df['Close'] + 1e-10),
        
        # 2. MQL5側でのインジケータハンドル宣言コード
        "mql5_handle": "hEMA_fast, hEMA_slow", # 複数ある場合はカンマ区切りまたは配列
        "mql5_handle_decls": "int hEMA_fast;\nint hEMA_slow;",
        
        # 3. MQL5側でのインジケータハンドル初期化コード
        "mql5_init": (
            "    hEMA_fast = iMA(smb, PERIOD_M5, 5, 0, MODE_EMA, PRICE_CLOSE);\n"
            "    hEMA_slow = iMA(smb, PERIOD_M5, 20, 0, MODE_EMA, PRICE_CLOSE);"
        ),
        
        # 4. MQL5側でのハンドル解放コード
        "mql5_release": "    IndicatorRelease(hEMA_fast);\n    IndicatorRelease(hEMA_slow);",
        
        # 5. MQL5側でのハンドルエラー検証コード
        "mql5_validation": "    if(hEMA_fast == INVALID_HANDLE || hEMA_slow == INVALID_HANDLE) return false;",
        
        # 6. MQL5側での特徴量配列コピー・計算コード (lag処理ループ内で使用されます)
        "mql5_calc": (
            "        double ema_f[1], ema_s[1];\n"
            "        if(CopyBuffer(hEMA_fast, 0, shift, 1, ema_f) < 1 || CopyBuffer(hEMA_slow, 0, shift, 1, ema_s) < 1) return false;\n"
            "        inputs[idx++] = (float)((ema_f[0] - ema_s[0]) / close_price);"
        )
    }
    
    # ※ 将来的に新しい特徴量をここへ追加していくことができます。
    # "New_Indicator": { ... }
}
