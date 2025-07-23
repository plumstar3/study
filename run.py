# run.py (論文の評価方法を完全に再現した最終版)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. データローダー関連
# ==============================================================================
def load_data(path='./Data/soybean_data.csv'):
    """CSVファイルを一度だけ読み込む。"""
    if not os.path.exists(path):
        print(f"エラー: データファイル '{path}' が見つかりません。")
        return None
    df = pd.read_csv(path)
    df = df[df['yield'] >= 5].reset_index(drop=True)
    return df

class SoybeanDataGenerator(tf.keras.utils.Sequence):
    """論文のコンセプトに基づき、地域ごとの5年シーケンスを生成するジェネレータ"""
    def __init__(self, df, batch_size, shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_cols = self.df.columns.drop(['loc_ID', 'year', 'yield'])
        
        self.loc_year_dict = { (row.loc_ID, int(row.year)): row for _, row in self.df.iterrows() }
        
        self.sequences = []
        loc_ids = self.df['loc_ID'].unique()
        all_years = sorted(self.df['year'].unique())

        target_years = sorted(self.df[self.df['year'] >= self.df['year'].min() + 4]['year'].unique())

        for loc_id in loc_ids:
            for target_year in target_years:
                seq_years = list(range(target_year - 4, target_year + 1))
                if all((loc_id, year) in self.loc_year_dict for year in seq_years):
                    self.sequences.append({'loc_id': loc_id, 'years': seq_years})
        
        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_seq_info = [self.sequences[i] for i in batch_indices]
        actual_batch_size = len(batch_seq_info)

        X_dict = {
            'e_input': np.zeros((actual_batch_size, 5, 312)),
            's_input': np.zeros((actual_batch_size, 5, 66)),
            'p_input': np.zeros((actual_batch_size, 5, 14)),
        }
        Y = np.zeros((actual_batch_size, 1))

        for i, seq_info in enumerate(batch_seq_info):
            loc_id, years = seq_info['loc_id'], seq_info['years']
            for j, year in enumerate(years):
                sample = self.loc_year_dict[(loc_id, year)]
                features = sample[self.feature_cols].values
                X_dict['e_input'][i, j, :] = features[0:312]
                X_dict['s_input'][i, j, :] = features[312:378]
                X_dict['p_input'][i, j, :] = features[378:392]
            Y[i] = self.loc_year_dict[(loc_id, years[-1])]['yield']
        return X_dict, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==============================================================================
# 2. モデル定義
# ==============================================================================
def create_cnn_block(input_layer, filters, kernel_sizes, name=""):
    x = input_layer
    for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
        x = layers.Conv1D(f, k, activation='relu', padding='same', name=f"{name}_conv_{i}")(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same', name=f"{name}_pool_{i}")(x)
    return layers.Flatten(name=f"{name}_flatten")(x)

def build_and_compile_model(sequence_length=5):
    e_input = layers.Input(shape=(sequence_length, 312), name="e_input")
    s_input = layers.Input(shape=(sequence_length, 66), name="s_input")
    p_input = layers.Input(shape=(sequence_length, 14), name="p_input")

    e_proc_input = layers.Input(shape=(312,), name="e_proc_input")
    e_reshaped = layers.Reshape((6, 52))(e_proc_input)
    shared_cnn_input = layers.Input(shape=(52, 1), name="shared_cnn_input")
    cnn_block_output = create_cnn_block(shared_cnn_input, [8, 16], [3, 3], name="shared_e_cnn")
    shared_e_cnn = models.Model(inputs=shared_cnn_input, outputs=cnn_block_output, name="Shared_E_CNN")
    e_cnn_outs = [shared_e_cnn(e_reshaped[:, i, :, None]) for i in range(6)]
    e_cnn_model = models.Model(inputs=e_proc_input, outputs=layers.Concatenate()(e_cnn_outs), name="E_CNN_Model")
    
    s_proc_input = layers.Input(shape=(66,), name="s_proc_input")
    s_reshaped = layers.Reshape((6, 11))(s_proc_input)
    s_cnn_out = create_cnn_block(s_reshaped, [16, 32], [3, 3], name="s_cnn")
    s_cnn_model = models.Model(inputs=s_proc_input, outputs=s_cnn_out, name="S_CNN_Model")

    e_processed = layers.TimeDistributed(e_cnn_model)(e_input)
    s_processed = layers.TimeDistributed(s_cnn_model)(s_input)
    p_processed = layers.TimeDistributed(layers.Flatten())(p_input)
    
    merged = layers.Concatenate()([e_processed, s_processed, p_processed])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.LSTM(64, return_sequences=False, dropout=0.2)(x)
    output = layers.Dense(1, name="Yhat1")(x)

    model = models.Model(inputs=[e_input, s_input, p_input], outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003), loss=losses.Huber(), metrics=['mae'])
    return model

# ==============================================================================
# 3. 訓練と評価のメイン実行関数
# ==============================================================================
def main():
    print("🌱 大豆収量予測モデル - 論文の評価方法を再現")
    data_path = './Data/soybean_data.csv'
    
    full_df = load_data(path=data_path)
    if full_df is None: return

    test_years = [2016, 2017, 2018]
    results = []
    
    for test_year in test_years:
        print("\n" + "="*50)
        print(f"🔬 テスト年: {test_year} での評価を開始")
        print("="*50)

        # 2. データ分割 (論文の記述に厳密に従う)
        train_df_unscaled = full_df[full_df['year'] < test_year].copy()
        test_df_unscaled = full_df[full_df['year'] == test_year].copy()
        
        if train_df_unscaled.empty or test_df_unscaled.empty:
            print("データ不足のためスキップ")
            continue

        # 3. 標準化 (訓練データの情報のみで学習)
        feature_cols = full_df.columns.drop(['loc_ID', 'year', 'yield'])
        scaler = StandardScaler()
        train_df = train_df_unscaled.copy(); train_df[feature_cols] = scaler.fit_transform(train_df_unscaled[feature_cols])
        test_df = test_df_unscaled.copy(); test_df[feature_cols] = scaler.transform(test_df_unscaled[feature_cols])
        
        # 4. データジェネレータの作成
        print("\n--- データ準備 ---")
        train_generator = SoybeanDataGenerator(train_df, batch_size=64)
        test_generator = SoybeanDataGenerator(pd.concat([train_df, test_df]), batch_size=64, shuffle=False)
        # 評価対象はテスト年のみにフィルタリング
        test_generator.sequences = [s for s in test_generator.sequences if s['years'][-1] == test_year]
        test_generator.indices = np.arange(len(test_generator.sequences))

        print(f"訓練用シーケンス数: {len(train_generator.sequences)}")
        print(f"テスト用シーケンス数 ({test_year}年): {len(test_generator.sequences)}")

        if len(train_generator) == 0 or len(test_generator) == 0:
            print("データ不足のためスキップ")
            continue
            
        # 5. モデルの構築と訓練
        model = build_and_compile_model()
        
        print("\n--- モデル訓練開始 ---")
        # 検証データは使わず、訓練データ全体で学習
        model.fit(train_generator, epochs=100, verbose=2,
                  callbacks=[callbacks.EarlyStopping(monitor='loss', patience=15)])
        print(f"✅ {test_year}年をテストケースとしたモデルの訓練完了")
        
        # 6. 評価
        print("\n--- モデル評価開始 ---")
        Y_pred = model.predict(test_generator).flatten()
        
        # 正解ラベルをジェネレータのシーケンス情報から直接取得
        true_yields = []
        for seq_info in test_generator.sequences:
            true_yields.append(test_generator.loc_year_dict[(seq_info['loc_id'], seq_info['years'][-1])]['yield'])
        Y_true = np.array(true_yields)

        rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
        results.append({'year': test_year, 'rmse': rmse})
        print(f"📊 {test_year}年のTest RMSE: {rmse:.4f}")

    # 7. 最終結果の表示
    print("\n" + "="*50)
    print("🎉 全てのクロスバリデーション評価が完了しました！")
    print("="*50)
    
    if results:
        results_df = pd.DataFrame(results).set_index('year')
        print(results_df)
        avg_rmse = results_df['rmse'].mean()
        print(f"\n  => 平均RMSE ({min(test_years)}-{max(test_years)}): {avg_rmse:.4f}")
    else:
        print("評価を実行できるデータがありませんでした。")

if __name__ == "__main__":
    main()