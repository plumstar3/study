import numpy as np

# ファイルのパス
file_path = 'prediction_result.npz'

try:
    # .npzファイルを読み込む
    data = np.load(file_path)

    # どんなデータが保存されているか名前一覧を表示
    print(f"📂 ファイル '{file_path}' に保存されているデータ名:")
    print(data.files)
    print("-" * 30)

    # それぞれのデータを変数に格納
    y_true = data['Y1_true']
    y_pred = data['Y1_pred']

    # 中身を表示
    print("🔬 正解の収量 (Y1_true):")
    print(y_true)
    print("\n🔬 モデルの予測収量 (Y1_pred):")
    print(y_pred)
    
    # 予測と正解を並べて比較
    print("\n--- 比較 ---")
    for true_val, pred_val in zip(y_true, y_pred):
        print(f"正解: {true_val[0]:.2f}  |  予測: {pred_val[0]:.2f}")


except FileNotFoundError:
    print(f"エラー: ファイル '{file_path}' が見つかりませんでした。")
except Exception as e:
    print(f"エラーが発生しました: {e}")