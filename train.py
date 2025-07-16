import tensorflow as tf
from tensorflow.keras import models, callbacks
from data_loader import load_and_preprocess_data, create_year_dict_and_avg, SoybeanDataGenerator
from model import build_and_compile_model # model.pyから直接インポート

def main():
    # 1. データのロードと前処理
    df = load_and_preprocess_data(path='./soybean_samples.csv')
    
    # 2. ジェネレータ用のデータ構造を作成
    year_dict, avg_dict = create_year_dict_and_avg(df)
    
    # 3. データジェネレータのインスタンス化 (ロジックはジェネレータクラス内に集約) ✨
    train_generator = SoybeanDataGenerator(df, year_dict, avg_dict, batch_size=25, is_training=True)
    val_generator = SoybeanDataGenerator(df, year_dict, avg_dict, batch_size=32, is_training=False)

    # 4. モデルの構築とコンパイル
    model = build_and_compile_model()
    model.summary()

    # 5. コールバックの設定
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # 6. 学習の実行
    # 検証データが1つもない場合は、検証をスキップする
    validation_steps = len(val_generator) if val_generator and len(val_generator) > 0 else None
    
    model.fit(
        train_generator,
        validation_data=val_generator if validation_steps else None,
        validation_steps=validation_steps,
        epochs=200,
        callbacks=[early_stop]
    )

    # 7. モデル保存
    model.save("soybean_yield_model.h5")
    print("✅ モデル学習完了・保存済み")

if __name__ == "__main__":
    main()