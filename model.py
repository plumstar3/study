import tensorflow as tf
from tensorflow.keras import layers, Model

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
    return Model(inputs=input_tensor, outputs=output_tensor, name=name)

def main_model(E_inputs, S_input, P_input, Ybar_input,
               num_units=64, num_layers=1, dropout_rate=0.0):
    
    shared_cnn_e = create_shared_cnn_E()

    e_outputs = [shared_cnn_e(inp) for inp in E_inputs]
    e_concat = layers.Concatenate()(e_outputs)
    e_dense = layers.Dense(40, activation='relu', name='e_dense')(e_concat)

    s_dense = layers.Dense(40, activation='relu', name='s_dense')(S_input)
    p_flat = layers.Flatten(name='p_flatten')(P_input)

    # 静的な特徴量を全て結合
    merged = layers.Concatenate()([e_dense, s_dense, p_flat])
    
    # ✨ Reshapeの代わりにRepeatVectorを使い、時系列データを作成
    x = layers.RepeatVector(5)(merged)
    
    # Ybar（年ごとの平均収量）を結合
    x = layers.Concatenate(axis=-1)([x, Ybar_input])

    # RNN (LSTM) 層
    for _ in range(num_layers):
        x = layers.LSTM(num_units, return_sequences=True, dropout=dropout_rate)(x)

    # TimeDistributedで各タイムステップにDenseを適用
    output = layers.TimeDistributed(layers.Dense(1))(x)

    # 最終時刻の予測(Yhat1)とそれ以前(Yhat2)を分離
    Yhat1 = layers.Lambda(lambda t: t[:, -1, :], name='Yhat1')(output)
    Yhat2 = layers.Lambda(lambda t: t[:, :-1, :], name='Yhat2')(output)

    return Yhat1, Yhat2