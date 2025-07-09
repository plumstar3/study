import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

# create_shared_cnn_E と main_model は変更なし
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
    # p_inputの形状が(None, 14)なので、Flattenは不要
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

# ▼▼▼ --- この関数を model.py の末尾に追加 --- ▼▼▼

def build_and_compile_model():
    """モデルの入力定義、構築、コンパイルをまとめて行う関数"""
    # 1. 入力層の定義
    E_inputs = [layers.Input(shape=(52, 1), name=f"E{i+1}") for i in range(6)]
    S_input = layers.Input(shape=(66,), name="S_input")
    P_input = layers.Input(shape=(14,), name="P_input")
    Ybar_input = layers.Input(shape=(5, 1), name="Ybar_input")

    all_inputs = E_inputs + [S_input, P_input, Ybar_input]

    # 2. モデル構築
    Yhat1, Yhat2 = main_model(E_inputs, S_input, P_input, Ybar_input)

    model = models.Model(inputs=all_inputs, outputs={'Yhat1': Yhat1, 'Yhat2': Yhat2})

    # 3. 損失関数と最適化手法を設定してコンパイル
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003),
                  loss={'Yhat1': 'mse', 'Yhat2': 'mse'},
                  loss_weights={'Yhat1': 1.0, 'Yhat2': 0.0},
                  metrics={'Yhat1': 'mae'})
    
    return model