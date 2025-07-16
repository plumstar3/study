# ==============================================================================
# 1. ライブラリのインポート
# ==============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


# ==============================================================================
# 2. データローダー関連 (元 data_loader.py)
# ==============================================================================
def load_and_preprocess_data(path='./soybean_samples.csv'):
    """
    CSVファイルを読み込み、基本的な前処理（標準化など）を行う。
    """
    df = pd.read_csv(path)
    
    # loc_ID, year, yield を除く特徴量部分を標準化
    feature_cols = df.columns[3:]
    
    # 訓練データ（2017年以前）で平均と標準偏差を計算
    train_df = df[df['year'] <= 2017]
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()
    
    # ゼロ除算を避けるため、標準偏差が0の列は1に置き換える
    std[std == 0] = 1.0

    # すべてのデータに標準化を適用
    df[feature_cols] = (df[feature_cols] - mean) / std
    
    # NaN値を0で埋める
    df = df.fillna(0)
    
    # 低収量データを除外
    df = df[df['yield'] >= 5].reset_index(drop=True)
    
    return df

def create_year_dict_and_avg(df):
    """
    年ごとのデータ辞書と、標準化された平均収量の辞書を作成する。
    """
    year_dict = {str(year): group for year, group in df.groupby('year')}
    
    avg_yield_by_year = df.groupby('year')['yield'].mean()
    
    mean_yield = avg_yield_by_year.mean()
    std_yield = avg_yield_by_year.std()
    avg_dict = (avg_yield_by_year - mean_yield) / std_yield
    
    if 2018 not in avg_dict.index and 2017 in avg_dict.index:
        avg_dict[2018] = avg_dict[2017]
        
    return year_dict, avg_dict.to_dict()

class SoybeanDataGenerator(tf.keras.utils.Sequence):
    """Kerasモデルのためのカスタムデータジェネレータ"""
    def __init__(self, df, year_dict, avg_dict, batch_size, is_training=True):
        self.df = df
        self.year_dict = year_dict
        self.avg_dict = avg_dict
        self.batch_size = batch_size
        self.is_training = is_training
        
        all_available_years = sorted(self.df['year'].unique())
        
        sequences = []
        for i in range(len(all_available_years) - 4):
            seq_candidate = all_available_years[i : i+5]
            if seq_candidate[-1] - seq_candidate[0] == 4:
                sequences.append(seq_candidate)

        sequences = np.array(sequences)

        if self.is_training:
            self.sequences = np.array([s for s in sequences if 2018 not in s])
        else:
            self.sequences = np.array([s for s in sequences if 2018 in s])
        
        print(f"{'訓練' if is_training else '検証'}ジェネレータが、{len(self.sequences)}個の有効な5年シーケンスを生成しました。")
        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        if len(self.sequences) == 0:
            return 0
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_seqs = self.sequences[batch_indices]
        
        actual_batch_size = len(batch_seqs)

        out_X = np.zeros((actual_batch_size, 5, 393))
        out_Y1 = np.zeros((actual_batch_size, 1))
        out_Y2 = np.zeros((actual_batch_size, 4, 1))

        for i, years in enumerate(batch_seqs):
            for j, year in enumerate(years):
                year_str = str(year)
                year_df = self.year_dict[year_str]
                avg_yield = self.avg_dict[year_str]
                sample = year_df.sample(1).iloc[0]
                features = sample.iloc[3:].values
                out_X[i, j, :] = np.concatenate([features, [avg_yield]])

            out_Y1[i] = sample['yield']
            past_years = years[:-1]
            past_yields = [self.year_dict[str(y)].sample(1).iloc[0]['yield'] for y in past_years]
            out_Y2[i] = np.array(past_yields).reshape(4, 1)

        X_dict, Y_dict = self._format_batch_for_model(out_X)
        
        Y_dict['Yhat1'] = out_Y1
        Y_dict['Yhat2'] = out_Y2

        return X_dict, Y_dict

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def _format_batch_for_model(self, batch_x):
        actual_batch_size = batch_x.shape[0]
        Ybar = batch_x[:, :, -1].reshape(actual_batch_size, 5, 1)
        
        inputs_dict = {
            **{f'E{i+1}': batch_x[:, i, 0:52].reshape(actual_batch_size, 52, 1) for i in range(5)},
            'E6': batch_x[:, 4, 52*5:52*6].reshape(actual_batch_size, 52, 1), 
            'S_input': batch_x[:, 0, 312:378],
            'P_input': batch_x[:, 0, 378:392],
            'Ybar_input': Ybar
        }
        
        return inputs_dict, {}


# ==============================================================================
# 3. モデル定義 (元 model.py)
# ==============================================================================
def create_shared_cnn_E(input_shape=(52, 1), name="Shared_CNN_E"):
    """気象データ(E)用の「共有」CNNモデルを作成する関数。"""
    input_tensor = layers.Input(shape=input_shape)
    x = layers.Conv1D(8, 9, activation='relu', padding='valid', kernel_initializer='glorot_uniform')(input_tensor)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(12, 3, activation='relu', padding='valid', kernel_initializer='glorot_uniform')(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(16, 3, activation='relu', padding='valid', kernel_initializer='glorot_uniform')(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(20, 3, activation='relu', padding='valid', kernel_initializer='glorot_uniform')(x)
    x = layers.AveragePooling1D(2)(x)
    output_tensor = layers.Flatten()(x)
    return models.Model(inputs=input_tensor, outputs=output_tensor, name=name)

def main_model(E_inputs, S_input, P_input, Ybar_input,
               num_units=64, num_layers=1, dropout_rate=0.0):
    
    shared_cnn_e = create_shared_cnn_E()

    e_outputs = [shared_cnn_e(inp) for inp in E_inputs]
    e_concat = layers.Concatenate()(e_outputs)
    e_dense = layers.Dense(40, activation='relu', name='e_dense')(e_concat)

    s_dense = layers.Dense(40, activation='relu', name='s_dense')(S_input)
    p_flat = P_input 

    merged = layers.Concatenate()([e_dense, s_dense, p_flat])
    x = layers.RepeatVector(5)(merged)
    x = layers.Concatenate(axis=-1)([x, Ybar_input])

    for _ in range(num_layers):
        x = layers.LSTM(num_units, return_sequences=True, dropout=dropout_rate)(x)

    output = layers.TimeDistributed(layers.Dense(1))(x)

    Yhat1 = layers.Lambda(lambda t: t[:, -1, :], name='Yhat1')(output)
    Yhat2 = layers.Lambda(lambda t: t[:, :-1, :], name='Yhat2')(output)

    return Yhat1, Yhat2

def build_and_compile_model():
    """モデルの入力定義、構築、コンパイルをまとめて行う関数"""
    E_inputs = [layers.Input(shape=(52, 1), name=f"E{i+1}") for i in range(6)]
    S_input = layers.Input(shape=(66,), name="S_input")
    P_input = layers.Input(shape=(14,), name="P_input")
    Ybar_input = layers.Input(shape=(5, 1), name="Ybar_input")

    all_inputs = E_inputs + [S_input, P_input, Ybar_input]

    Yhat1, Yhat2 = main_model(E_inputs, S_input, P_input, Ybar_input)

    model = models.Model(inputs=all_inputs, outputs={'Yhat1': Yhat1, 'Yhat2': Yhat2})

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003),
                  loss={'Yhat1': 'mse', 'Yhat2': 'mse'},
                  loss_weights={'Yhat1': 1.0, 'Yhat2': 0.0},
                  metrics={'Yhat1': 'mae'})
    
    return model


# ==============================================================================
# 4. 訓練・評価・実行 (元 train.py, evaluate.py, main.py)
# ==============================================================================
def run_training():
    """モデルの訓練を実行する関数"""
    # 1. データのロードと前処理
    df = load_and_preprocess_data(path='./soybean_samples.csv')
    
    # 2. ジェネレータ用のデータ構造を作成
    year_dict, avg_dict = create_year_dict_and_avg(df)
    
    # 3. データジェネレータのインスタンス化
    train_generator = SoybeanDataGenerator(df, year_dict, avg_dict, batch_size=25, is_training=True)
    val_generator = SoybeanDataGenerator(df, year_dict, avg_dict, batch_size=32, is_training=False)

    # 4. モデルの構築とコンパイル
    model = build_and_compile_model()
    model.summary()

    # 5. コールバックの設定
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # 6. 学習の実行
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

def run_evaluation(model_path="soybean_yield_model.h5"):
    """学習済みモデルの評価を実行する関数"""
    # 1. データのロードと前処理
    df = load_and_preprocess_data(path='./soybean_samples.csv')
    year_dict, avg_dict = create_year_dict_and_avg(df)
    
    # 2. 評価用データをジェネレータで準備
    test_generator = SoybeanDataGenerator(df, year_dict, avg_dict, batch_size=32, is_training=False) # バッチサイズは任意
    
    if len(test_generator) == 0:
        print("評価データがありません。評価をスキップします。")
        return

    # 3. ジェネレータからテストデータを取得
    X_test, Y_test = test_generator[0]
    Y1_test_true = Y_test['Yhat1']

    # 4. モデルの読み込みと予測
    model = models.load_model(model_path)
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
    print("🌱 大豆収量予測モデル - 総合実行スクリプト")
    
    print("\n🛠️ モデルの訓練を開始します...")
    run_training()
    
    print("\n🔍 モデルの評価を開始します...")
    run_evaluation()
    
    print("\n🎉 全処理が完了しました！")