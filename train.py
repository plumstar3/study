import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from data_loader import load_and_preprocess_data, create_year_dict_and_avg, SoybeanDataGenerator
from model import main_model
import numpy as np

def build_and_compile_model():
    """モデルの入力定義、構築、コンパイルを行う"""
    # 1. 入力定義 (新しいデータ構造に合わせる) ✨
    E_inputs = [layers.Input(shape=(52, 1), name=f"E{i+1}") for i in range(6)]
    S_input = layers.Input(shape=(66,), name="S_input")
    P_input = layers.Input(shape=(14,), name="P_input")
    Ybar_input = layers.Input(shape=(5, 1), name="Ybar_input")

    all_inputs = E_inputs + [S_input, P_input, Ybar_input]

    # 2. モデル構築
    Yhat1, Yhat2 = main_model(E_inputs, S_input, P_input, Ybar_input)

    model = models.Model(inputs=all_inputs, outputs={'Yhat1': Yhat1, 'Yhat2': Yhat2})

    # 3. 損失関数・最適化
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003), # 元のコードの学習率に合わせる
                  loss={'Yhat1': 'mse', 'Yhat2': 'mse'},
                  loss_weights={'Yhat1': 1.0, 'Yhat2': 0.0}, # 元のコードの重みに合わせる
                  metrics={'Yhat1': 'mae'})
    
    return model

def main():
    # 1. データのロードと基本前処理
    df = load_and_preprocess_data(path='./soybean_samples.csv')
    
    # 2. ジェネレータ用のデータ構造を作成
    year_dict, avg_dict = create_year_dict_and_avg(df)
    
    # 5年単位の年のシーケンスを作成
    year_sequences = np.array([np.arange(year - 4, year + 1) for year in range(1984, 2019)])
    
    # 3. 訓練/検証シーケンスの分割
    train_sequences = np.array([seq for seq in year_sequences if 2018 not in seq])
    val_sequences = np.array([seq for seq in year_sequences if 2018 in seq])

    # 4. データジェネレータのインスタンス化
    train_generator = SoybeanDataGenerator(df, train_sequences, year_dict, avg_dict, batch_size=25)
    val_generator = SoybeanDataGenerator(df, val_sequences, year_dict, avg_dict, batch_size=len(val_sequences))

    # 5. モデルの構築とコンパイル
    model = build_and_compile_model()
    model.summary()

    # 6. コールバックの設定
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # 7. 学習の実行
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=200, # エポック数を調整
        callbacks=[early_stop],
        workers=4,
        use_multiprocessing=True
    )

    # 8. モデル保存
    model.save("soybean_yield_model.h5")
    print("✅ モデル学習完了・保存済み")

if __name__ == "__main__":
    main()