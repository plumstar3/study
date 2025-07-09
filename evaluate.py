import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from data_loader import load_and_preprocess_data, create_year_dict_and_avg, SoybeanDataGenerator

def evaluate_model(model_path="soybean_yield_model.h5"):
    # 1. データのロードと前処理
    df = load_and_preprocess_data(path='./soybean_samples.csv')
    year_dict, avg_dict = create_year_dict_and_avg(df)
    year_sequences = np.array([np.arange(year - 4, year + 1) for year in range(1984, 2019)])
    
    # 2. 評価用データ（2018年）をジェネレータで準備
    test_sequences = np.array([seq for seq in year_sequences if 2018 in seq])
    test_generator = SoybeanDataGenerator(
        df, 
        test_sequences, 
        year_dict, 
        avg_dict, 
        batch_size=len(test_sequences)
    )
    
    # 3. ジェネレータからテストデータを取得
    X_test, Y_test = test_generator[0]
    Y1_test_true = Y_test['Yhat1']

    # 4. モデルの読み込みと予測
    model = load_model(model_path)
    print("✅ モデル読み込み完了")
    
    Y_pred = model.predict(X_test)
    Y1_pred = Y_pred['Yhat1']

    # 5. 評価指標の計算
    rmse = np.sqrt(mean_squared_error(Y1_test_true, Y1_pred))
    print(f"📊 Test RMSE (final year): {rmse:.4f}")

    corr, _ = pearsonr(Y1_test_true.flatten(), Y1_pred.flatten())
    print(f"📈 相関係数 (final year): {corr:.4f}")

    np.savez("prediction_result.npz", Y1_true=Y1_test_true, Y1_pred=Y1_pred)
    print("📝 予測結果を 'prediction_result.npz' に保存しました")

if __name__ == "__main__":
    evaluate_model()