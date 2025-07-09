from train import main as run_training
from evaluate import evaluate_model

if __name__ == "__main__":
    print("🌱 大豆収量予測モデル - 総合実行スクリプト")
    
    print("\n🛠️ モデルの訓練を開始します...")
    run_training()
    print("✅ モデル訓練完了！")

    print("\n🔍 モデルの評価を開始します...")
    evaluate_model()
    print("✅ モデル評価完了！")

    print("\n🎉 全処理が完了しました！")