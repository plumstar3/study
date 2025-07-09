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
    
    # 低収量データを除外 (元のロジックを維持)
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
    
    if 2018 not in avg_dict:
        avg_dict[2018] = avg_dict[2017]
        
    return year_dict, avg_dict.to_dict()

class SoybeanDataGenerator(tf.keras.utils.Sequence):
    """Kerasモデルのためのカスタムデータジェネレータ"""
    def __init__(self, df, year_sequences, year_dict, avg_dict, batch_size):
        self.df = df
        self.year_sequences = year_sequences
        self.year_dict = year_dict
        self.avg_dict = avg_dict
        self.batch_size = batch_size
        self.indices = np.arange(len(self.year_sequences))
        self.on_epoch_end()

    def __len__(self):
        """1エポックあたりのステップ数を返す"""
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """1バッチ分のデータを生成して返す"""
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_seqs = self.year_sequences[batch_indices]
        
        # 5年 x 393特徴量 (W:312 + S:66 + P:14 + Ybar:1)
        out_X = np.zeros((self.batch_size, 5, 393))
        out_Y1 = np.zeros((self.batch_size, 1))
        out_Y2 = np.zeros((self.batch_size, 4, 1))

        for i, years in enumerate(batch_seqs):
            for j, year in enumerate(years):
                year_df = self.year_dict[str(year)]
                avg_yield = self.avg_dict[str(year)]
                
                # その年のデータからランダムに1サンプル選択
                sample = year_df.sample(1).iloc[0]
                
                features = sample.iloc[3:].values # ID, year, yield を除く
                out_X[i, j, :] = np.concatenate([features, [avg_yield]])

            out_Y1[i] = sample['yield']
            
            # Y2の教師データ（過去4年分の実績収量）を取得
            past_years = years[:-1]
            past_yields = [self.year_dict[str(y)].sample(1).iloc[0]['yield'] for y in past_years]
            out_Y2[i] = np.array(past_yields).reshape(4, 1)

        X_dict, Y_dict = self._format_batch_for_model(out_X, out_Y1, out_Y2)
        
        return X_dict, Y_dict

    def on_epoch_end(self):
        """エポック終了時にインデックスをシャッフルする"""
        np.random.shuffle(self.indices)

    def _format_batch_for_model(self, batch_x, batch_y1, batch_y2):
        """バッチデータをモデル入力用の辞書形式に変換する"""
        Ybar = batch_x[:, :, -1].reshape(self.batch_size, 5, 1)
        
        inputs_dict = {
            **{f'E{i+1}': batch_x[:, i, 0:52].reshape(-1, 52, 1) for i in range(5)},
            'E6': batch_x[:, 4, 52:104].reshape(-1, 52, 1), # 6番目のデータがないので5番目を仮に流用
            'S_input': batch_x[:, 0, 312:378],
            'P_input': batch_x[:, 0, 378:392],
            'Ybar_input': Ybar
        }
        
        outputs_dict = {'Yhat1': batch_y1, 'Yhat2': batch_y2}

        return inputs_dict, outputs_dict