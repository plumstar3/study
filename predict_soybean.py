import numpy as np
import time
import tensorflow as tf

# TF 2.x では eager execution がデフォルトなので、v1互換モードは不要

class ConvResPartP(tf.keras.layers.Layer):
    def __init__(self, var_name, **kwargs):
        super(ConvResPartP, self).__init__(name=var_name + '_P', **kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        P_t = inputs
        X = self.flatten(P_t)
        # print('conv2 out P', X.shape) # .shape を追加
        return X

class ConvResPartE(tf.keras.layers.Layer):
    def __init__(self, var_name, **kwargs):
        super(ConvResPartE, self).__init__(name=var_name + '_E', **kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='valid',
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=None,
                                            name='Conv00_' + var_name)
        self.relu1 = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)

        self.conv2 = tf.keras.layers.Conv1D(filters=12, kernel_size=3, strides=1, padding='valid',
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=None,
                                            name='Conv0_' + var_name)
        self.relu2 = tf.keras.layers.ReLU()
        self.pool2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)

        self.conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='valid',
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=None,
                                            name='Conv1_' + var_name)
        self.relu3 = tf.keras.layers.ReLU()
        self.pool3 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)

        self.conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=3, strides=1, padding='valid', # s0 は 1
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=None,
                                            name='Conv2_' + var_name)
        self.relu4 = tf.keras.layers.ReLU()
        self.pool4 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()


    def call(self, inputs, training=None): # is_training は training に変更
        E_t = inputs
        X = self.conv1(E_t)
        X = self.relu1(X)
        X = self.pool1(X)
        X = self.conv2(X)
        X = self.relu2(X)
        X = self.pool2(X)
        X = self.conv3(X)
        X = self.relu3(X)
        X = self.pool3(X)
        # print('conv1 out E', X.shape) # .shape を追加
        X = self.conv4(X)
        X = self.relu4(X)
        X = self.pool4(X)
        # print('E outttt', X.shape) # .shape を追加
        X = self.flatten(X)
        return X

class ConvResPartS(tf.keras.layers.Layer):
    def __init__(self, var_name, **kwargs):
        super(ConvResPartS, self).__init__(name=var_name + '_S', **kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters=4, kernel_size=3, strides=1, padding='valid',
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=None,
                                            name='Conv1S_' + var_name)
        self.relu1 = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)

        self.conv2 = tf.keras.layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='valid',
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=None,
                                            name='Conv2S_' + var_name)
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv1D(filters=12, kernel_size=2, strides=1, padding='valid',
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(), activation=None,
                                            name='Conv3S_' + var_name)
        self.relu3 = tf.keras.layers.ReLU()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None): # is_training は training に変更
        S_t = inputs
        X = self.conv1(S_t)
        X = self.relu1(X)
        X = self.pool1(X)
        X = self.conv2(X)
        X = self.relu2(X)
        X = self.conv3(X)
        X = self.relu3(X)
        X = self.flatten(X)
        return X


class CNNRNNModel(tf.keras.Model):
    def __init__(self, num_units, num_layers, dropout_rate, f_val, **kwargs): # dropout を dropout_rate に変更
        super(CNNRNNModel, self).__init__(**kwargs)
        self.f_val = f_val # f は使用されていないように見えるが、念のため保持

        # E parts (共有ウェイトのつもりであれば、同じインスタンスを呼び出す)
        # 元のコードでは var_name='v1' で共有しているため、同じインスタンスを定義
        self.conv_e_shared = ConvResPartE(var_name='v1_e_shared')

        # S parts (共有ウェイトのつもりであれば、同じインスタンスを呼び出す)
        self.conv_s_shared = ConvResPartS(var_name='v1_s_shared')

        self.conv_p = ConvResPartP(var_name='p_part')

        self.e_fc = tf.keras.layers.Dense(units=40, activation='relu',
                                          kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                          bias_initializer=tf.zeros_initializer())
        self.s_fc = tf.keras.layers.Dense(units=40, activation='relu', # 元は activation=None の後 relu
                                           kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                           bias_initializer=tf.zeros_initializer())
        
        # LSTM layers
        # 元のコードでは num_layers=1 なので、1つのLSTMレイヤー
        self.lstm_cells = []
        for _ in range(num_layers): # num_layers が1より大きい場合に対応
             # Dropout は Keras LSTM レイヤーの引数で指定
            self.lstm_cells.append(tf.keras.layers.LSTMCell(num_units, dropout=dropout_rate))
        
        # MultiRNNCell に相当する StackedRNNCells を使用
        # num_layers=1 の場合は tf.keras.layers.LSTM で直接可
        if num_layers == 1:
             self.lstm_layer = tf.keras.layers.LSTM(num_units, return_sequences=True, dropout=dropout_rate)
        else:
            # StackedRNNCells を tf.keras.layers.RNN でラップ
            stacked_lstm_cells = tf.keras.layers.StackedRNNCells(self.lstm_cells)
            self.lstm_layer = tf.keras.layers.RNN(stacked_lstm_cells, return_sequences=True)


        self.final_fc = tf.keras.layers.Dense(units=1, activation=None,
                                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                              bias_initializer=tf.zeros_initializer())

    def call(self, inputs, training=None):
        E_t_list, S_t_list, P_t, Ybar, S_t_extra = inputs
        # E_t_list は [E_t1, E_t2, ..., E_t6]
        # S_t_list は [S_t1, S_t2, ..., S_t10]

        e_outputs = [self.conv_e_shared(e_input, training=training) for e_input in E_t_list]
        e_out_concat = tf.concat(e_outputs, axis=1)
        # print('after E concatenate', e_out_concat.shape)
        e_out_fc = self.e_fc(e_out_concat)
        # print('e_out_fc_*************', e_out_fc.shape)

        s_outputs = [self.conv_s_shared(s_input, training=training) for s_input in S_t_list]
        s_out_concat = tf.concat(s_outputs, axis=1)
        # print('soil after S concatenate', s_out_concat.shape)
        s_out_fc = self.s_fc(s_out_concat)
        # print('soil after FC layer', s_out_fc.shape)

        p_out = self.conv_p(P_t) # training 引数は ConvResPartP にはない
        # print('p outtttttt', p_out.shape)

        combined_out = tf.concat([e_out_fc, s_out_fc, p_out], axis=1)
        # print('soil + Weather after concatante', combined_out.shape)

        time_step = 5 # 元のコードから
        # combined_out の最後の次元を計算
        # (batch_size, features) -> (batch_size, time_step, features_per_step)
        # combined_out の特徴量が time_step で割り切れる必要がある
        # 元のコード: e_out=tf.reshape(e_out,shape=[-1,time_step,e_out.get_shape().as_list()[-1]])
        # ここでは combined_out.shape[1] が e_out.get_shape().as_list()[-1] に相当
        # ただし、Ybar と S_t_extra を連結する前の e_out なので注意
        # このreshapeは入力データの準備方法に依存する。
        # 元の main_proccess では、このreshapeの後に Ybar と S_t_extra を連結している。
        # 構造を合わせるため、 Ybar と S_t_extra の連結前にreshapeが必要か、
        # あるいはreshape後の次元と Ybar, S_t_extra の次元を合わせる必要がある。
        
        # 元のコードのreshapeの意図を再現:
        # e_out の次元は (batch_size, num_features_concat)
        # これを (batch_size, time_step, features_per_time_step) にする
        # features_per_time_step = num_features_concat / time_step
        
        num_features_concat = combined_out.shape[1]
        if num_features_concat is None: # Symbolic tensor
            # This path might be taken during model building before shapes are fully known.
            # For dynamic shapes, tf.shape can be used.
            # However, for tf.keras.layers.Reshape, the target shape needs to be defined.
            # If running eagerly, .shape will be concrete.
            # Assuming feature dimension is known or can be inferred for Reshape layer.
            # For now, let's assume it will be known during actual execution.
             pass # Or handle symbolic shape
        
        # Keras Reshape Layer を使用
        # (バッチサイズ, 全特徴量) -> (バッチサイズ, タイムステップ, 全特徴量/タイムステップ)
        # Reshapeのためには、reshape後の各次元の積がreshape前の次元の積と一致する必要がある。
        # ここでの `combined_out` は1Dの特徴量ベクトル（バッチごと）。
        # LSTMに入力するためには (batch, timesteps, feature) の形式が必要。
        # `time_step` は5とハードコードされている。
        # `combined_out` を (batch_size * time_step, features_for_lstm_per_step) のような形にするか、
        # もともとのデータ準備で (batch_size, time_step, features) の形にするのが一般的。
        
        # 元のコード: e_out=tf.reshape(e_out,shape=[-1,time_step,e_out.get_shape().as_list()[-1]])
        # これは間違っていて、最後の次元は features_per_step であるべき。
        # e_out.get_shape().as_list()[-1] は、reshape前のconcatされた特徴量の総数。
        # 正しくは features_per_step = combined_out.shape[1] // time_step

        # データの準備段階で時間軸の情報を保持し、それをLSTMに入力するのが一般的
        # ここでは元のコードのreshape操作を模倣するが、データの持ち方に注意が必要
        
        # この時点での combined_out は (batch_size, total_features)
        # Ybar (batch, 5, 1), S_t_extra (batch, 5, 4) (元のplaceholderから推定、ただしS_t_extraのreshape後は(?,5,4))
        # LSTM入力は (batch_size, time_step, features)
        # combined_out を (batch_size, time_step, features_per_step) に変形する必要がある
        
        # 簡略化のため、combined_outがすでに (batch_size, time_step, features_per_step_cnn) の形になっていると仮定するか、
        # もしくは (batch_size, total_cnn_features) を (batch_size, 1, total_cnn_features) のようにして、
        # それを Ybar, S_t_extra と時間軸で concat する前に time_step 分繰り返すなどの処理が必要。
        
        # 元コードの main_proccess の reshape と concat の順序を確認:
        # 1. e_out = concat([e_fc, s_fc, p_out])  # Shape: (batch, combined_features)
        # 2. e_out = tf.reshape(e_out, shape=[-1, time_step, e_out.get_shape().as_list()[-1]])
        #    -> この reshape は次元がおかしい。正しくは features_per_step = combined_features / time_step
        #       shape=[-1, time_step, features_per_step]
        # 3. S_t_extra = tf.reshape(S_t_extra, shape=[-1, time_step, 4])
        # 4. e_out = tf.concat([e_out, Ybar, S_t_extra], axis=-1)
        
        # 修正案:
        features_per_step = combined_out.shape[1] // time_step
        lstm_input_cnn = tf.reshape(combined_out, shape=[-1, time_step, features_per_step])
        # print('lstm_input_cnn after reshape', lstm_input_cnn.shape)

        # Ybar と S_t_extra の形状を合わせる (元は placeholder で time_step=5 を想定しているように見える)
        # Ybar: (batch, time_step, 1)
        # S_t_extra: (batch, time_step, 4)
        # print("Ybar shape before LSTM concat:", Ybar.shape)
        # print("S_t_extra shape before LSTM concat:", S_t_extra.shape)

        lstm_input = tf.concat([lstm_input_cnn, Ybar, S_t_extra], axis=-1)
        # print('lstm_input after concat', lstm_input.shape)

        lstm_output = self.lstm_layer(lstm_input, training=training)
        # print('RNN output shape', lstm_output.shape) # (batch, time_step, num_units)

        # 元のコードでは reshape して FC に入れている
        # output=tf.reshape(output,shape=[-1,output.get_shape().as_list()[-1]])
        # -> (batch * time_step, num_units)
        output_reshaped = tf.reshape(lstm_output, shape=[-1, lstm_output.shape[-1]])
        
        output_fc = self.final_fc(output_reshaped)
        # print("final_fc output shape", output_fc.shape) # (batch * time_step, 1)

        # 元のコード: output = tf.reshape(output, shape=[-1,5]) # 5はtime_step
        # -> (batch, time_step)  (最後の次元が1なのでsqueezeされていると仮定)
        final_output_reshaped = tf.reshape(output_fc, shape=[-1, time_step])
        # print("output of all time steps", final_output_reshaped.shape)

        # Yhat1: 最後のタイムステップの予測
        Yhat1 = tf.expand_dims(final_output_reshaped[:, 4], axis=1) # 0-indexedなので4
        # print('Yhat1 shape', Yhat1.shape)

        # Yhat2: 最後の手前のタイムステップの予測
        Yhat2 = final_output_reshaped[:, 0:4]
        # print('Yhat2 shape', Yhat2.shape)

        return Yhat1, Yhat2

def cost_function_tf2(Y, Yhat): # 名前変更
    huber_loss = tf.keras.losses.Huber(delta=5.0) # delta は元のコードから
    Loss = huber_loss(Y, Yhat)
    
    E = Y - Yhat
    E2 = tf.pow(E, 2)
    MSE = tf.reduce_mean(E2) # 元のコードでは squeeze していたが、reduce_meanでスカラーになる
    RMSE = tf.pow(MSE, 0.5)
    return RMSE, MSE, E, Loss


# get_sample と get_sample_te は numpy 操作なので変更なし
def get_sample(dic,L,avg,batch_size,time_steps,num_features):
    L_tr=L[:-1,:]
    out=np.zeros(shape=[batch_size,time_steps,num_features])
    for i in range(batch_size):
        r1 = np.squeeze(np.random.randint(L_tr.shape[0], size=1))
        years = L_tr[r1, :]
        for j, y in enumerate(years):
            X_year_data = dic[str(y)] # X は予約語なので変更
            ym=avg[str(y)]
            r2 = np.random.randint(X_year_data.shape[0], size=1)
            out[i, j, :] = np.concatenate((X_year_data[r2, :],np.array([[ym]])),axis=1)
    return out

def get_sample_te(dic,mean_last,avg,batch_size_te,time_steps,num_features):
    out = np.zeros(shape=[batch_size_te, time_steps, num_features])
    X_year_data = dic[str(2018)] # X は予約語なので変更
    # mean_last の reshape の次元数を確認
    # 元のコード: mean_last.reshape(1,4,3+6*52+1+100+14+4)
    # num_features = 316+100+14+4 = 434
    # mean_last は (4 * (元の特徴量数)) のはず。元の特徴量数は 3(ID,year,yield) + 434 -1(avg_yield) = 436 ?
    # この部分は元のデータの準備方法に依存するため、そのままでは動かない可能性が高い
    # mean_last の生成ロジックと get_sample_te の reshape を整合させる必要がある
    # ここでは num_features - 1 (avg_yield分を引く) で reshape する想定
    original_feature_dim = num_features -1 # avg[str(y)] の分
    
    # mean_last は4つの年の平均データ (各年は original_feature_dim)
    # out[:, 0:4, :-1] に代入する
    if mean_last.size == 4 * original_feature_dim : # サイズが合うか確認
         out[:, 0:4, :-1] = mean_last.reshape(1, 4, original_feature_dim)
    else:
        print(f"Warning: mean_last size {mean_last.size} does not match expected {4 * original_feature_dim}")
        # fallback or error handling
    
    ym=np.zeros(shape=[batch_size_te,1])+avg['2018']
    # X_year_data は (num_samples_2018, original_feature_dim)のはず
    # batch_size_te 分だけランダムサンプリングするか、全て使うか。元は全てX[r1,:]だったがr1は使われていない
    # ここでは batch_size_te > X_year_data.shape[0] の場合に問題が起きる
    # 元のコードでは X_year_data をそのまま使っている (暗黙的に batch_size_te と X_year_data.shape[0] が一致する想定？)
    # np.sum(Index) が batch_size_te になるので、X_year_data.shape[0] と一致するはず
    out[:,4,:-1]=X_year_data[:,:] # X全体を使う
    out[:,4,-1]=ym.squeeze() # avg yield を最後の特徴量として追加
    return out


def main_program(X_data_full, Index_val, num_units, num_layers, Max_it, learning_rate_init, batch_size_tr, le, l_const, num_features): # X, Index, learning_rate, l は変数名変更
    
    model = CNNRNNModel(num_units=num_units, num_layers=num_layers, dropout_rate=0.0, f_val=3) # dropout は training で制御, fは元のコードから
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_init)

    # TensorBoard writer
    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = './logs/gradient_tape/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    t1=time.time()
    A = []
    for i in range(4, 39):
        A.append([ i - 4, i - 3, i - 2, i - 1, i])
    A = np.vstack(A)
    A += 1980
    # print(A.shape)

    dic = {}
    for i in range(39): # 0 to 38, for years 1980 to 2017
        year_str = str(i + 1980)
        dic[year_str] = X_data_full[X_data_full[:, 1] == (i + 1980)]

    # 2018年のデータを辞書に追加 (get_sample_te で使われるため)
    dic['2018'] = X_data_full[X_data_full[:, 1] == 2018]


    avg = {}
    avg2 = []
    for i in range(39):
        year_str = str(i + 1980)
        avg[year_str] = np.mean(X_data_full[X_data_full[:, 1] == (i + 1980)][:, 2])
        avg2.append(avg[year_str])

    # print('avgggggg raw', avg)
    mm = np.mean(avg2)
    ss = np.std(avg2)

    for i in range(39):
        year_str = str(i + 1980)
        avg[year_str] = (np.mean(X_data_full[X_data_full[:, 1] == (i + 1980)][:, 2]) - mm) / ss
    
    avg['2018'] = avg.get(str(2017), 0) # 2017年がない場合のエラーを防ぐ (通常はあるはず)
    # print('avgggggg scaled', avg)

    # mean_last の生成 (過去4年間の平均的な特徴量)
    # 元のデータの次元: LocationID, Year, Yield, Weather(52*6), Soil(100), Planting(14), Soil_extra(4)
    # Total features in X_data_full[:, 3:] = 312 + 100 + 14 + 4 = 430
    # get_sample/get_sample_te の num_features = 430 + 1 (avg_yield) = 431
    # dic[year_str] は (num_samples, 3 + 430) の形状
    # X_data_full[X_data_full[:, 1] == year] は (num_samples_for_year, 3 + 430 features)
    # dic[str(year)] = X[X[:, 1] == i + 1980]  -> この X は X_data_full のこと
    # np.mean(dic['2014'], axis=0) は (1, 3 + 430)
    # これに avg['2014'] (スカラー) を連結するので、次元が合わない。
    # 元のコードでは、おそらく X[r2, :] の部分でスライスされた後のデータ (特徴量のみ) を想定している。
    # get_sample の out[i,j,:] = np.concatenate((X_year_data[r2, :],np.array([[ym]])),axis=1)
    # X_year_data[r2, :] は (1, 3 + 430)のはず。ymはスカラー。
    # np.array([[ym]]) は (1,1)。これを axis=1 で連結すると (1, 3 + 430 + 1)
    # num_features はこの連結後の特徴量数なので、3+430+1 = 434
    # mean_last は、この434次元の特徴量を過去4年分について平均し、それを連結したもの。
    
    # mean_last 生成ロジック修正
    mean_features_list = []
    for year_val in [2014, 2015, 2016, 2017]:
        year_str = str(year_val)
        if year_str in dic and dic[year_str].shape[0] > 0:
            avg_yield_for_year = np.full((dic[year_str].shape[0], 1), avg[year_str])
            data_with_avg_yield = np.concatenate((dic[year_str], avg_yield_for_year), axis=1) # dic[year_str]を使用
            mean_feature_for_year = np.mean(data_with_avg_yield, axis=0)
            mean_features_list.append(mean_feature_for_year)
        else:
            print(f"Warning: No data for year {year_str} in mean_last calculation.")
            # num_features を使用
            mean_features_list.append(np.zeros(num_features)) # num_total_features_in_dic_plus_avg ではなく num_features

    if len(mean_features_list) == 4:
        mean_last = np.concatenate(mean_features_list, axis=0)
    else:
        print("Error: Could not generate mean_last correctly.")
        mean_last = np.zeros((4, num_features)) # num_features を使用

    validation_loss_hist = [] # 名前変更
    train_loss_hist = []    # 名前変更
    
    current_lr = learning_rate_init

    for i in range(Max_it):
        out_tr = get_sample(dic, A, avg, batch_size_tr, time_steps=5, num_features=num_features) # num_features は434

        # データをモデルの入力形式に整形
        # E_t1..6, S_t1..10, P_t, Ybar, S_t_extra
        # out_tr は (batch, time_step, num_features=434)
        # num_features は [loc_id, year, yield, W(52*6), S(100), P(14), S_extra(4), avg_yield_scaled(1)]
        # 合計: 3 + 312 + 100 + 14 + 4 + 1 = 434
        
        # 元のfeed_dictのBatch_X_eのスライスを参考に入力を作成
        # Batch_X_e = out_tr[:, :, 3:-1].reshape(-1, 6*52+100+14+4)
        # つまり、loc_id, year, yield を除き、最後のavg_yieldも除く
        # 特徴量部分: out_tr[:, :, 3:-1] -> (batch, time_step, 430)
        
        # 各タイムステップのデータを分離
        all_step_features = out_tr[:, :, 3:-1] # (batch, 5, 430)
        
        E_t_all_steps = [all_step_features[:, :, 0:52],      # E_t1相当 (バッチの全タイムステップ分)
                         all_step_features[:, :, 52*1:52*2], # E_t2相当
                         all_step_features[:, :, 52*2:52*3], # E_t3相当
                         all_step_features[:, :, 52*3:52*4], # E_t4相当
                         all_step_features[:, :, 52*4:52*5], # E_t5相当
                         all_step_features[:, :, 52*5:52*6]] # E_t6相当
        # 各E_ti は (batch, time_step, 52)。モデル入力は (batch, features) なので、時間軸をどう扱うか？
        # 元のコードはplaceholder (None, 52, 1) だった。これは (batch, features, channels)
        # CNN1Dの入力は (batch, steps, channels) または (batch, features, channels)
        # ここでは (batch, num_weeks=52, num_env_features_per_week=1) と解釈
        # E_t_list に入れるのは (batch_size, 52, 1) の形式のテンソルを6つ。
        # out_tr からスライスしたデータは (batch_size, time_step, 52)
        # これを (batch_size * time_step, 52, 1) にして入力するか、
        # モデル側で時間軸を扱うか。
        # 元の main_proccess の呼び出しでは E_t1 などは時間軸の情報を持たず、
        # LSTM の前で reshape して時間軸を作っていた。
        # よって、CNN部分は時間軸なしの特徴量 (batch, features, 1) を受け取る。
        # Batch_X_e は (-1, 430, 1) の形にしていた。
        # この reshape は (batch_size * time_step, feature_dim_per_step, 1)
        
        # 訓練データの準備
        # Ybar_tr = out_tr[:, :, -1].reshape(-1, 5, 1) # (batch, time_step, 1)
        # Batch_Y = out_tr[:, -1, 2].reshape(-1, 1) # (batch, 1) - 最後のタイムステップの実際の収量
        # Batch_Y_2 = out_tr[:, 0:4, 2] # (batch, 4) - それ以前のタイムステップの実際の収量

        # 元の feed_dict のスライスに合わせて入力データを作成
        # Batch_X_e = out_tr[:, :, 3:-1] # (batch, 5, 430 features)
        # これを (batch * 5, 430, 1) にreshapeし、そこからスライス
        # X_e_for_feed = np.expand_dims(out_tr[:, :, 3:-1].reshape(-1, 430), axis=-1)
        
        # モデル入力は時間軸ごとに独立した特徴量を想定している (LSTM前で時間軸再構成)
        # よって、各E_ti, S_ti, P_t は (batch_size, feature_dim, 1)
        # out_tr の各タイムステップのデータをバッチ次元に展開して処理するか、
        # 1サンプル (5タイムステップ) を1つのバッチとして扱うか。
        # 元のコードの batch_size_tr はサンプル数なので、後者。
        # E_t1, S_t1 などは (batch_tr, feature_dim, 1)
        
        # ここでは1つのタイムステップのデータのみをCNNに通し、その結果を時間軸方向にconcatする方針でモデルが書かれている。
        # e_out1 = conv_res_part_E(E_t1, ...)
        # E_t1 は (batch, 52, 1)
        # このデータ準備が複雑。元の get_sample は (batch, time_steps, num_total_features_incl_id_yield_avgyield)
        
        # 簡略化と元の構造の尊重：
        # モデルの入力は E_t_list, S_t_list, P_t, Ybar, S_t_extra
        # 各 E_t_i は (batch_size, 52, 1)
        # 各 S_t_i は (batch_size, 10, 1)
        # P_t は (batch_size, 14, 1)
        # Ybar は (batch_size, 5, 1)
        # S_t_extra は (batch_size, 5, 4) (reshape後)
        
        # out_tr: (batch_size_tr, 5タイムステップ, 434特徴量)
        # 434特徴量 = [id, year, yield, W(312), S(100), P(14), S_extra(4), avg_yield(1)]
        
        # 1. Ybar_tr と Y_t, Y_t_2 を抽出
        Ybar_tr_feed = out_tr[:, :, -1].astype(np.float32).reshape(batch_size_tr, 5, 1)
        Y_t_feed = out_tr[:, 4, 2].astype(np.float32).reshape(batch_size_tr, 1) # last time step yield
        Y_t_2_feed = out_tr[:, 0:4, 2].astype(np.float32) # previous time step yields
        
        # 2. 特徴量を抽出 (W, S, P, S_extra)
        # これらは時間軸ごとに異なる値を持つのではなく、1つのサンプルに対して1セット与えられ、
        # それがCNN処理され、その結果が時間軸方向に並べられてLSTMに入る構造。
        # よって、out_tr のどのタイムステップの特徴量を使うか？
        # 元のコードでは Batch_X_e は out_tr[:, :, 3:-1].reshape(-1, ...) していた。
        # これは全タイムステップの特徴量をフラットにしていたように見える。
        # そして、そのフラットなものから E_t1 等をスライス。これは時間情報を混ぜている。
        
        # 想定される構造：
        # 1サンプル (e.g., 1 location-year のシーケンス) が out_tr の1行 (axis 0) に対応。
        # このサンプルは5年分のデータ (time_steps=5)。
        # ただし、CNNへの入力となる E_t1..6, S_t1..10, P_t は「現在の予測対象年」に関する静的な特徴量のはず。
        # おそらく、out_tr の最後のタイムステップ (予測年) の特徴量を使う。
        current_time_step_features = out_tr[:, 4, 3:-1].astype(np.float32) # (batch, 430)
        
        E_feed_list = []
        current_idx = 0
        for _ in range(6): # E_t1 to E_t6
            E_feed_list.append(np.expand_dims(current_time_step_features[:, current_idx : current_idx+52], axis=-1))
            current_idx += 52
            
        # current_time_step_features の形状確認 (バッチサイズ, 430 を期待)
        print(f"Debug: Shape of current_time_step_features before S_feed_list: {current_time_step_features.shape}")
        
        S_feed_list = []
        # current_idx の初期値確認 (312 を期待)
        print(f"Debug: Initial current_idx for S_feed_list: {current_idx}")
        
        for k_s in range(10): # S_t1 to S_t10
            start_idx_s = current_idx
            end_idx_s = current_idx + 10
            
            print(f"Debug: S_feed_list loop {k_s+1}, Slicing current_time_step_features with indices {start_idx_s}:{end_idx_s}")
            
            s_slice = current_time_step_features[:, start_idx_s : end_idx_s]
            # s_slice の形状確認 (バッチサイズ, 10 を期待)
            print(f"Debug:   Shape of s_slice for S_t{k_s+1}: {s_slice.shape}")
            
            if s_slice.shape[1] == 0:
                print(f"ERROR: s_slice for S_t{k_s+1} has 0 features! This will cause a crash.")
                # ここで詳細なデバッグ情報をさらに表示するか、例外を発生させることも検討できます。
                # 例: print(f"       current_idx: {current_idx}, total features in current_time_step_features: {current_time_step_features.shape[1]}")

            expanded_s_slice = np.expand_dims(s_slice, axis=-1)
            # expanded_s_slice の形状確認 (バッチサイズ, 10, 1 を期待)
            # print(f"Debug:   Shape of expanded_s_slice for S_t{k_s+1}: {expanded_s_slice.shape}")
            
            S_feed_list.append(expanded_s_slice.astype(np.float32)) # .astype(np.float32) をここで適用
            current_idx += 10
        
        # P_t_feed と S_t_extra_feed の準備も同様に current_idx を使用
        print(f"Debug: current_idx before P_t_feed: {current_idx}") # 412 を期待
        P_t_feed = np.expand_dims(current_time_step_features[:, current_idx : current_idx+14], axis=-1).astype(np.float32)
        print(f"Debug:   Shape of P_t_feed: {P_t_feed.shape}") # (バッチサイズ, 14, 1) を期待
        current_idx += 14
        
        print(f"Debug: current_idx before S_t_extra_flat: {current_idx}") # 426 を期待
        S_t_extra_flat = current_time_step_features[:, current_idx : current_idx+4]
        print(f"Debug:   Shape of S_t_extra_flat: {S_t_extra_flat.shape}") # (バッチサイズ, 4) を期待
        S_t_extra_feed = np.tile(np.expand_dims(S_t_extra_flat, axis=1), (1, 5, 1)).astype(np.float32)


        if i == 60000:
            current_lr = current_lr / 2
            optimizer.learning_rate.assign(current_lr)
            # print('learningrate1', current_lr)
        elif i == 120000:
            current_lr = current_lr / 2
            optimizer.learning_rate.assign(current_lr)
            # print('learningrate2', current_lr)
        elif i == 180000:
            current_lr = current_lr / 2
            optimizer.learning_rate.assign(current_lr)
            # print('learningrate3', current_lr)

        with tf.GradientTape() as tape:
            # モデル呼び出し
            # model_inputs = (E_feed_list, S_feed_list, P_t_feed, Ybar_tr_feed, S_t_extra_feed)
            # 修正: E_feed_listの各要素は(batch, 52, 1)など、S_feed_listも同様。
            # P_t_feed (batch, 14, 1), Ybar_tr_feed (batch, 5, 1), S_t_extra_feed (batch, 5, 4)
            yhat1_pred, yhat2_pred = model((E_feed_list, S_feed_list, P_t_feed, Ybar_tr_feed, S_t_extra_feed), training=True)
            
            _, _, _, loss1 = cost_function_tf2(Y_t_feed, yhat1_pred)
            _, _, _, loss2 = cost_function_tf2(Y_t_2_feed, yhat2_pred) # Y_t_2_feed は (batch,4), yhat2_pred も (batch,4)
            
            total_loss = tf.constant(l_const, dtype=tf.float32) * loss1 + tf.constant(le, dtype=tf.float32) * loss2
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i % 1000 == 0:
            # Training metrics
            rmse_tr_val, _, _, loss_tr_val = cost_function_tf2(Y_t_feed, yhat1_pred)
            rc_tr = np.corrcoef(np.squeeze(Y_t_feed), np.squeeze(yhat1_pred.numpy()))[0, 1] # .numpy() でTF TensorからNumpyへ
            train_loss_hist.append(loss_tr_val.numpy())

            # Validation
            out_te = get_sample_te(dic, mean_last, avg, np.sum(Index_val), time_steps=5, num_features=num_features) # num_features=434
            
            Ybar_te_feed = out_te[:, :, -1].astype(np.float32).reshape(np.sum(Index_val), 5, 1)
            Y_te_feed = out_te[:, 4, 2].astype(np.float32).reshape(np.sum(Index_val), 1)
            Y_te_2_feed = out_te[:, 0:4, 2].astype(np.float32)
            
            current_time_step_features_te = out_te[:, 4, 3:-1].astype(np.float32)
            E_te_feed_list = []
            current_idx_te = 0
            for _ in range(6):
                E_te_feed_list.append(np.expand_dims(current_time_step_features_te[:, current_idx_te : current_idx_te+52], axis=-1))
                current_idx_te += 52
            S_te_feed_list = []
            for _ in range(10):
                S_te_feed_list.append(np.expand_dims(current_time_step_features_te[:, current_idx_te : current_idx_te+10], axis=-1))
                current_idx_te += 10
            P_te_t_feed = np.expand_dims(current_time_step_features_te[:, current_idx_te : current_idx_te+14], axis=-1)
            current_idx_te += 14
            S_te_t_extra_flat = current_time_step_features_te[:, current_idx_te : current_idx_te+4]
            S_te_t_extra_feed = np.tile(np.expand_dims(S_te_t_extra_flat, axis=1), (1, 5, 1)).astype(np.float32)

            yhat1_te_pred, _ = model((E_te_feed_list, S_te_feed_list, P_te_t_feed, Ybar_te_feed, S_te_t_extra_feed), training=False)
            rmse_te_val, _, _, loss_val_val = cost_function_tf2(Y_te_feed, yhat1_te_pred)
            rc_te = np.corrcoef(np.squeeze(Y_te_feed), np.squeeze(yhat1_te_pred.numpy()))[0, 1]
            validation_loss_hist.append(loss_val_val.numpy())

            print(f"Iteration {i}, Train RMSE: {rmse_tr_val.numpy():.4f}, Cor: {rc_tr:.4f}, Test RMSE: {rmse_te_val.numpy():.4f}, Cor: {rc_te:.4f}")
            print(f"Train Loss: {loss_tr_val.numpy():.4f}, Val Loss: {loss_val_val.numpy():.4f}")
            
            with summary_writer.as_default():
                tf.summary.scalar('train_rmse', rmse_tr_val, step=i)
                tf.summary.scalar('train_loss_total', loss_tr_val, step=i)
                tf.summary.scalar('val_rmse', rmse_te_val, step=i)
                tf.summary.scalar('val_loss_total', loss_val_val, step=i)
                tf.summary.scalar('learning_rate', current_lr, step=i)


    # Final evaluation on test set (similar to validation block)
    out_te_final = get_sample_te(dic, mean_last, avg, np.sum(Index_val), time_steps=5, num_features=num_features)
    Ybar_te_final_feed = out_te_final[:, :, -1].astype(np.float32).reshape(np.sum(Index_val), 5, 1)
    Y_te_final_feed = out_te_final[:, 4, 2].astype(np.float32).reshape(np.sum(Index_val), 1)
    # Y_te_2_final_feed = out_te_final[:, 0:4, 2].astype(np.float32) # Not used for final RMSE calculation in original
    
    current_time_step_features_te_final = out_te_final[:, 4, 3:-1].astype(np.float32)
    E_te_final_feed_list = []
    current_idx_te_final = 0
    for _ in range(6):
        E_te_final_feed_list.append(np.expand_dims(current_time_step_features_te_final[:, current_idx_te_final : current_idx_te_final+52], axis=-1))
        current_idx_te_final += 52
    S_te_final_feed_list = []
    for _ in range(10):
        S_te_final_feed_list.append(np.expand_dims(current_time_step_features_te_final[:, current_idx_te_final : current_idx_te_final+10], axis=-1))
        current_idx_te_final += 10
    P_te_t_final_feed = np.expand_dims(current_time_step_features_te_final[:, current_idx_te_final : current_idx_te_final+14], axis=-1)
    current_idx_te_final += 14
    S_te_t_extra_final_flat = current_time_step_features_te_final[:, current_idx_te_final : current_idx_te_final+4]
    S_te_t_extra_final_feed = np.tile(np.expand_dims(S_te_t_extra_final_flat, axis=1), (1, 5, 1)).astype(np.float32)

    yhat1_te_final_pred, _ = model((E_te_final_feed_list, S_te_final_feed_list, P_te_t_final_feed, Ybar_te_final_feed, S_te_t_extra_final_feed), training=False)
    rmse_te_final_val, _, _, _ = cost_function_tf2(Y_te_final_feed, yhat1_te_final_pred)

    print(f"The final training RMSE was (approx last reported): {rmse_tr_val.numpy():.4f} and test RMSE is {rmse_te_final_val.numpy():.4f}")
    t2 = time.time()
    print(f'The training time was {round(t2 - t1, 2):.2f}')
    
    # Save model
    model.save_weights('./model_soybean_tf2_weights', save_format='tf') # または .h5
    # print(f"Total trainable parameters: {np.sum([np.prod(v.shape) for v in model.trainable_variables])}")


    return rmse_tr_val.numpy(), rmse_te_final_val.numpy(), train_loss_hist, validation_loss_hist


if __name__ == '__main__':
    BigX = np.load('./Soybeans_Data.npz')
    X_data_full = BigX['data'] # Renamed X to X_data_full

    # Normalize features (cols 3 onwards)
    X_tr_features = X_data_full[X_data_full[:, 1] <= 2017][:, 3:]
    M = np.mean(X_tr_features, axis=0, keepdims=True)
    S = np.std(X_tr_features, axis=0, keepdims=True)
    S[S == 0] = 1e-8 # Avoid division by zero if std is zero for some feature
    X_data_full[:, 3:] = (X_data_full[:, 3:] - M) / S
    X_data_full = np.nan_to_num(X_data_full)

    # Filter low yield (same as original)
    index_low_yield = X_data_full[:, 2] < 5
    print('low yield observations', np.sum(index_low_yield))
    # print(X_data_full[index_low_yield][:, 1])
    X_data_full = X_data_full[np.logical_not(index_low_yield)]
    del BigX

    Index_val_year = X_data_full[:, 1] == 2018  # validation year (boolean index)

    print('Std %.2f and mean %.2f of test ' % (np.std(X_data_full[Index_val_year][:, 2]), np.mean(X_data_full[Index_val_year][:, 2])))
    print("train data (before 2018)", np.sum(X_data_full[:, 1] < 2018)) # More precise
    print("test data (2018)", np.sum(Index_val_year))

    # num_features for get_sample functions
    # [loc_id, year, yield, W(52*6), S(100), P(14), S_extra(4), avg_yield_scaled(1)]
    # 3 + 312 + 100 + 14 + 4 + 1 = 434
    num_total_features_in_dic_plus_avg = X_data_full.shape[1] + 1


    Max_it_param = 350000      # 150000 could also be used with early stopping
    learning_rate_param = 0.0003   # Learning rate
    batch_size_tr_param = 25  # traning batch size
    le_param = 0.0  # Weight of loss for prediction using times before final time steps
    l_param = 1.0    # Weight of loss for prediction using final time step
    num_units_param = 64  # Number of hidden units for LSTM celss
    num_layers_param = 1  # Number of layers of LSTM cell

    # Run main program
    rmse_tr, rmse_te, train_loss, validation_loss = main_program(
        X_data_full, Index_val_year, num_units_param, num_layers_param,
        Max_it_param, learning_rate_param, batch_size_tr_param, le_param, l_param,
        num_total_features_in_dic_plus_avg
    )
    # print("Final Train RMSE:", rmse_tr)
    # print("Final Test RMSE:", rmse_te)
    # Plotting losses (optional)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,5))
    # plt.plot(train_loss, label='Train Loss')
    # plt.plot(validation_loss, label='Validation Loss')
    # plt.xlabel('Iterations (x1000)')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Loss')
    # plt.savefig('loss_plot_tf2.png')
    # plt.show()