# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.layers import Bidirectional


# 1. データローダー関連
def load_and_preprocess_data(path='./Data/soybean_data.csv'):
    """CSVファイルを読み込み、基本的な前処理（標準化など）を行う。"""
    if not os.path.exists(path):
        print(f"エラー: データファイル '{path}' が見つかりません。プログラムを終了します。")
        return None
    df = pd.read_csv(path)
    
    feature_cols = df.columns[3:]
    train_df = df[df['year'] <= 2017]
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()
    std[std == 0] = 1.0
    df[feature_cols] = (df[feature_cols] - mean) / std
    df = df.fillna(0)
    df = df[df['yield'] >= 5].reset_index(drop=True)
    return df

def create_year_loc_dict_and_avg(df):
    """年と地域(loc_ID)をキーにしたデータ辞書と、年ごとの平均収量辞書を作成する。"""
    loc_year_dict = { (row.loc_ID, int(row.year)): row for index, row in df.iterrows() }
    avg_yield_by_year = df.groupby('year')['yield'].mean()
    mean_yield = avg_yield_by_year.mean()
    std_yield = avg_yield_by_year.std()
    avg_dict = (avg_yield_by_year - mean_yield) / std_yield
    
    if 2018 not in avg_dict.index and 2017 in avg_dict.index:
        avg_dict[2018] = avg_dict.get(2017, 0)
        
    return loc_year_dict, {str(k): v for k, v in avg_dict.to_dict().items()}

class SoybeanDataGenerator(tf.keras.utils.Sequence):
    """Kerasモデルのためのカスタムデータジェネレータ（地域考慮・TimeDistributed対応版）"""
    def __init__(self, df, loc_year_dict, avg_dict, batch_size, is_training=True, target_year=None):
        self.loc_year_dict = loc_year_dict
        self.avg_dict = avg_dict
        self.batch_size = batch_size
        self.target_year = target_year
        self.is_training = is_training
        
        self.sequences = []
        loc_ids = df['loc_ID'].unique()
        all_years = sorted(df['year'].unique())

        for loc_id in loc_ids:
            for i in range(len(all_years) - 4):
                seq_years = all_years[i:i+5]
                if all((loc_id, year) in self.loc_year_dict for year in seq_years):
                    self.sequences.append({'loc_id': loc_id, 'years': seq_years})

        # --- 対象年を含む or 含まない でフィルタ ---
        if target_year is not None:
            if is_training:
                self.sequences = [s for s in self.sequences if target_year not in s['years']]
            else:
                self.sequences = [s for s in self.sequences if target_year in s['years']]
        
        print(f"{'訓練' if is_training else '検証'}ジェネレータが、{len(self.sequences)}個の有効な「地域-5年」シーケンスを生成しました。")
        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        if len(self.sequences) == 0: return 0
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_seq_info = [self.sequences[i] for i in batch_indices]
        actual_batch_size = len(batch_seq_info)

        #入力データを辞書形式で準備
        X_dict = {
            'e_input': np.zeros((actual_batch_size, 5, 312)),
            's_input': np.zeros((actual_batch_size, 5, 66)),
            'p_input': np.zeros((actual_batch_size, 5, 14)),
            'ybar_input': np.zeros((actual_batch_size, 5, 1))
        }
        Y_dict = {
            'Yhat1': np.zeros((actual_batch_size, 1)),
            'Yhat2': np.zeros((actual_batch_size, 4, 1))
        }

        for i, seq_info in enumerate(batch_seq_info):
            loc_id = seq_info['loc_id']
            years = seq_info['years']
            
            for j, year in enumerate(years):
                sample = self.loc_year_dict[(loc_id, year)]
                features = sample.iloc[3:].values # ID, year, yieldを除く
                
                # 特徴量を各入力に割り当て
                X_dict['e_input'][i, j, :] = features[0:312]
                X_dict['s_input'][i, j, :] = features[312:378]
                X_dict['p_input'][i, j, :] = features[378:392]
                X_dict['ybar_input'][i, j, 0] = self.avg_dict[str(year)]

            Y_dict['Yhat1'][i] = self.loc_year_dict[(loc_id, years[-1])]['yield']
            past_yields = [self.loc_year_dict[(loc_id, y)]['yield'] for y in years[:-1]]
            Y_dict['Yhat2'][i] = np.array(past_yields).reshape(4, 1)

        return X_dict, Y_dict

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# 2. モデル定義
def create_cnn_block(input_layer, filters, kernel_sizes):
    """汎用的なCNNブロックを作成"""
    x = input_layer
    for f, k in zip(filters, kernel_sizes):
        x = layers.Conv1D(f, k, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x) # ストライドを調整
    return layers.Flatten()(x)

def build_and_compile_model():
    """
    論文の設計思想（重み共有とTimeDistributed）を忠実に再現したモデルを構築する。
    """
    # --- 入力層の定義 (年ごとの変化を捉えるため、5つのタイムステップを持つ) ---
    e_input = layers.Input(shape=(5, 312), name="e_input")
    s_input = layers.Input(shape=(5, 66), name="s_input")
    p_input = layers.Input(shape=(5, 14), name="p_input")
    ybar_input = layers.Input(shape=(5, 1), name="ybar_input")

    # --- 特徴量処理ブロックの定義 (サブモデルとして) ---
    # ✨【修正点1】: 6種類の気象データ全てに適用する「共有CNNブロック」を1つだけ定義
    e_cnn_input = layers.Input(shape=(52, 1), name="e_cnn_input")
    x = layers.Conv1D(8, 9, activation='relu', padding='valid')(e_cnn_input)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(12, 3, activation='relu', padding='valid')(x)
    x = layers.AveragePooling1D(2)(x)
    e_cnn_output = layers.Flatten()(x)
    shared_e_cnn = models.Model(inputs=e_cnn_input, outputs=e_cnn_output, name="Shared_E_CNN")

    # ✨【修正点2】: 6つの気象データを処理するためのラッパーモデルを定義
    e_proc_input = layers.Input(shape=(312,), name="e_proc_input")
    e_reshaped = layers.Reshape((6, 52, 1))(e_proc_input)
    # 6つの入力それぞれに、上で定義した「全く同じ共有CNN」を適用する
    e_sub_outputs = [shared_e_cnn(e_reshaped[:, i]) for i in range(6)]
    e_proc_output = layers.Concatenate()(e_sub_outputs)
    e_processor = models.Model(inputs=e_proc_input, outputs=e_proc_output, name="E_Processor")

    # 土壌(S)データ用CNNモデル
    s_proc_input = layers.Input(shape=(66,), name="s_proc_input")
    s_reshaped = layers.Reshape((6, 11))(s_proc_input)
    s_cnn_out = layers.Flatten()(layers.Conv1D(16, 3, activation='relu')(s_reshaped))
    s_processor = models.Model(inputs=s_proc_input, outputs=s_cnn_out, name="S_Processor")

    # --- TimeDistributedで各タイムステップに特徴量処理を適用 ---
    e_processed = layers.TimeDistributed(e_processor, name="TDD_E_Processor")(e_input)
    s_processed = layers.TimeDistributed(s_processor, name="TDD_S_Processor")(s_input)
    p_processed = layers.TimeDistributed(layers.Flatten(), name="TDD_P_Flatten")(p_input)

    # --- 全ての特徴量を結合し、LSTMに入力 ---
    merged = layers.Concatenate()([e_processed, s_processed, p_processed, ybar_input])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    output = layers.TimeDistributed(layers.Dense(1))(x)
    
    Yhat1 = layers.Identity(name='Yhat1')(output[:, -1, :])
    Yhat2 = layers.Identity(name='Yhat2')(output[:, :-1, :])

    model = models.Model(inputs=[e_input, s_input, p_input, ybar_input], outputs=[Yhat1, Yhat2])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003),
                  loss={'Yhat1': losses.Huber(), 'Yhat2': losses.Huber()},
                  loss_weights={'Yhat1': 1.0, 'Yhat2': 0.0},
                  metrics={'Yhat1': 'mae'})
    return model

# 3. 訓練と評価（論文準拠: 2016, 2017, 2018年を検証年として個別に評価）
def run_training_and_evaluation():
    print("\n[モデル評価] 論文方式（2016〜2018年を検証年とする）で訓練・評価を開始します。")
    df = load_and_preprocess_data()
    if df is None:
        return
    
    check_years = list(range(2012, 2017))
    valid_locs = []

    for loc_id in df['loc_ID'].unique():
        years = df[df['loc_ID'] == loc_id]['year'].unique()
        if all(y in years for y in check_years):
            valid_locs.append(loc_id)

    print(f"検証年2016のための有効なloc_ID数: {len(valid_locs)}")

    loc_year_dict, avg_dict = create_year_loc_dict_and_avg(df)
    
    results = {}

    for test_year in [2016, 2017, 2018]:
        print(f"\n=== 評価対象年: {test_year} ===")

        # --- 学習・検証用データフレームの作成 ---
        train_df = df[df['year'] < test_year].copy()
        val_df = df[df['year'].between(test_year - 4, test_year)].copy()

        # --- ジェネレータ作成 ---
        train_generator = SoybeanDataGenerator(train_df, loc_year_dict, avg_dict, batch_size=32, is_training=True, target_year=test_year)
        val_generator = SoybeanDataGenerator(val_df, loc_year_dict, avg_dict, batch_size=26, is_training=False, target_year=test_year)

        if len(train_generator) == 0 or len(val_generator) == 0:
            print(f"⚠️ 学習 or 検証データが不足しているため {test_year}年はスキップされました。")
            continue

        # --- モデル構築・訓練 ---
        model = build_and_compile_model()
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=200,
            callbacks=[early_stop],
            verbose=2
        )

        # --- 評価 ---
        print(f"\n▶ モデル評価（{test_year}）")
        predictions = model.predict(val_generator)
        Y1_pred = predictions[0]
        Y1_true = np.concatenate([val_generator[i][1]['Yhat1'] for i in range(len(val_generator))])

        rmse = np.sqrt(mean_squared_error(Y1_true, Y1_pred))
        corr, _ = pearsonr(Y1_true.flatten(), Y1_pred.flatten())

        results[test_year] = {'rmse': rmse, 'corr': corr}
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - 相関係数: {corr:.4f}")

        np.savez(f"prediction_{test_year}.npz", Y1_true=Y1_true, Y1_pred=Y1_pred)
        print(f"📁 予測結果を prediction_{test_year}.npz に保存しました")

    # --- 総合結果表示 ---
    print("\n=== 総合評価結果 ===")
    for year in sorted(results.keys()):
        print(f"  {year}: RMSE={results[year]['rmse']:.4f}, 相関係数={results[year]['corr']:.4f}")

# 4. メイン実行ブロック
if __name__ == "__main__":
    print(" 大豆収量予測モデル - 総合実行スクリプト")
    run_training_and_evaluation()
    print("\n 全処理が完了しました！")