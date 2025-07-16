import numpy as np
import tensorflow as tf
import pandas as pd

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
    
    # 低収量データを除外 (この処理が特定の年を丸ごと削除する原因になりうる)
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
    
    # .indexを使って安全に存在確認を行う
    if 2018 not in avg_dict.index and 2017 in avg_dict.index:
        avg_dict[2018] = avg_dict[2017]
        
    
    return year_dict, {str(k): v for k, v in avg_dict.to_dict().items()}

class SoybeanDataGenerator(tf.keras.utils.Sequence):
    """Kerasモデルのためのカスタムデータジェネレータ（最終版）"""
    def __init__(self, df, year_dict, avg_dict, batch_size, is_training=True):
        self.df = df
        self.year_dict = year_dict
        self.avg_dict = avg_dict
        self.batch_size = batch_size
        self.is_training = is_training
        
        # ✨【重要】コンストラクタで「有効な」シーケンスのみを安全に生成する
        # データフレームから、前処理後に実際に存在する年のみを取得
        all_available_years = sorted(self.df['year'].unique())
        
        sequences = []
        # 存在する年のリストから、5年連続のシーケンスのみを抽出
        for i in range(len(all_available_years) - 4):
            seq_candidate = all_available_years[i : i+5]
            # 実際に5年連続しているかを確認 (例: 1985-1980 == 5 ではない)
            if seq_candidate[-1] - seq_candidate[0] == 4:
                sequences.append(seq_candidate)

        sequences = np.array(sequences)

        # 訓練用と検証用にシーケンスを分割
        if self.is_training:
            self.sequences = np.array([s for s in sequences if 2018 not in s])
        else:
            self.sequences = np.array([s for s in sequences if 2018 in s])
        
        # デバッグ用：生成されたシーケンスの数を出力
        print(f"{'訓練' if is_training else '検証'}ジェネレータが、{len(self.sequences)}個の有効な5年シーケンスを生成しました。")

        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        # バッチが1つもない場合でも0を返すようにする
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
                features = sample.iloc[3:].values # ID, year, yield を除く
                out_X[i, j, :] = np.concatenate([features, [avg_yield]])

            # 最後の年の実際の収量をY1の教師データとする
            out_Y1[i] = sample['yield']
            
            # Y2の教師データ（過去4年分の実績収量）を取得
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
            # 5つのタイムステップの気象データをループで生成
            **{f'E{i+1}': batch_x[:, i, 0:52].reshape(actual_batch_size, 52, 1) for i in range(5)},
            # 6番目の気象データは、CSVの W_6_x に対応
            'E6': batch_x[:, 4, 52*5:52*6].reshape(actual_batch_size, 52, 1), 
            'S_input': batch_x[:, 0, 312:378],
            'P_input': batch_x[:, 0, 378:392],
            'Ybar_input': Ybar
        }
        
        return inputs_dict, {}