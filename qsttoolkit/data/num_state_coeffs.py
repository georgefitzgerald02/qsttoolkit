import math
import numpy as np


coeffs17 = [
    [
        1 / math.sqrt(6) * math.sqrt(7 - math.sqrt(17)), 
        0, 
        0, 
        1 / math.sqrt(6) * math.sqrt(math.sqrt(17) - 1), 
        0
    ],
    [
        0, 
        1 / math.sqrt(6) * math.sqrt(9 - math.sqrt(17)), 
        0, 
        0, 
        1 / math.sqrt(6) * math.sqrt(math.sqrt(17) - 3)
    ]
]

coeffsM = [
    [0.545835, -3.7726e-9, 4.84951e-8, -0.711441, -7.48481e-8, -1.3146e-8, 0.441725, 1.15458e-8, 1.06094e-8, -0.0281825, -6.02332e-9, -6.39204e-9, 0.000376419, -6.91869e-9],
    [2.48927e-9, -0.744685, -8.04083e-9, 6.01943e-8, -0.570602, -3.1519e-8, -7.38494e-10, -0.346003, -8.48565e-9, -1.21143e-8, 0.0117984, -4.66046e-9, -5.09037e-9, -0.000107586]
]

coeffsP = [
    [0, 0.756, 0, 0, -0.515, -0.208, 0.127, 0.051, 0.317],
    [-0.558, 0, 0, -0.701, -0.0572, 0, -0.275, -0.333, 0.0794]
]

coeffsP2 = [
    [-0.504, 0.0837, -0.225, 0, -0.453, -0.523, 0.252, 0, 0.0955, 0.217, 0, 0, 0, -0.279, -0.0827, -0.051],
    [0, 0.501, 0.484, -0.387, 0.0566, -0.256, -0.0903, -0.189, 0.108, -0.2, -0.418, -0.0574, 0, 0, -0.142, 0]
]

coeffsM2 = [
    [
        -0.45717455741713664, -1.0856965103853774e-6 + 1.3239037829080093e-6j, -0.35772784377291084 - 0.048007740168066144j,
        -3.5459165445315755e-6 + 1.2571453643232864e-5j, -0.5383420820794502 - 0.24179040513272307j,
        9.675641330014822e-7 + 4.569566899500361e-6j, 0.2587482691377581 + 0.313044506480362j,
        4.1979351791851435e-6 - 1.122460690803522e-6j, -0.11094500303308243 + 0.20905585817734396j,
        -1.1837814323046472e-6 + 3.8758497675466054e-7j, 0.1275629945870373 - 0.1177987279989385j,
        -2.690647673469878e-6 - 3.6519804939862998e-6j, 0.12095531973074151 - 0.19588735180644176j,
        -2.6588791126371675e-6 - 6.058292629669095e-7j, 0.052905370429015865 - 0.0626791930782206j,
        -1.6615538648519722e-7 + 6.756126951837809e-8j, 0.016378329200891946 - 0.034743342821208854j,
        4.408946495377283e-8 + 2.2826415255126898e-8j, 0.002765352838800482 - 0.010624191776867055j,
        6.429253878486627e-8, 0.00027095836439738105 - 0.002684435917226972j,
        1.1081202749445256e-8 - 2.938812506852636e-8j, -0.000055767533641099717 - 0.000525444354381421j,
        -1.0776974926155464e-8 - 2.497769263148397e-8j, -0.000024992489351114305 - 0.00008178444317382933j,
        -1.5079116121444066e-8 - 2.0513760149701907e-8j, -5.64035228941742e-6 - 1.0297667130821428e-5j,
        -1.488452012610573e-8 - 1.7358623165948514e-8j, -8.909884885392901e-7 - 1.04267002748775e-6j,
        -1.2056784102984098e-8 - 1.2210951690230782e-8j
    ],
    [
        0, 0.5871298855433338, -3.3729618710801137e-6 + 2.4152360811650373e-6j, -0.5233926069798007 - 0.13655786303346068j,
        -4.623380373113224e-6 + 1.0362902695259763e-5j, -0.17909656013941788 - 0.11916639160269833j,
        -3.399720873431807e-6 - 7.125008373682292e-7j, 0.04072119358712736 - 0.3719310475303641j,
        -7.536125619789242e-6 + 1.885248226837573e-6j, -0.11393851510585044 - 0.3456924286310791j,
        -2.3915763815197452e-6 - 4.2406689395594674e-7j, 0.12820184730203607 + 0.0935942533049232j,
        -1.5407293261691393e-6 - 2.4673669087089514e-6j, -0.012272903377715643 - 0.13317144020065683j,
        -1.1260776123106269e-6 - 1.6865728072273087e-7j, -0.01013345155253134 - 0.0240812705564227j,
        0 - 1.4163391111474348e-7j, -0.003213070562510137 - 0.012363639898516247j,
        -1.0619280312362908e-8 - 1.2021213613319027e-7j, -0.002006756716685063 - 0.0026636832583059812j,
        0 - 4.509035934797572e-8j, -0.00048585160444833446 - 0.0005014735884977489j,
        -1.2286988061034212e-8 - 2.1199721851825594e-8j, -0.00010897007463988193 - 0.00007018240288615613j,
        -1.2811279935244964e-8 - 1.160553871672415e-8j, -0.00001785800494916693 - 6.603027186486886e-6j,
        -1.1639448324793031e-8, -2.4097385882316104e-6 - 3.5223103057306496e-7j,
        -1.0792272866841885e-8, -2.597671478115077e-7 + 2.622928060603902e-8j
    ]
]


states17 = np.array(coeffs17)
states17 = states17.reshape(states17.shape[0], states17.shape[1], 1)
statesM = np.array(coeffsM)
statesM = statesM.reshape(statesM.shape[0], statesM.shape[1], 1)
statesP = np.array(coeffsP)
statesP = statesP.reshape(statesP.shape[0], statesP.shape[1], 1)
statesP2 = np.array(coeffsP2)
statesP2 = statesP2.reshape(statesP2.shape[0], statesP2.shape[1], 1)
statesM2 = np.array(coeffsM2)
statesM2 = statesM2.reshape(statesM2.shape[0], statesM2.shape[1], 1)


num_state_params = [1.5615528128088303, 2.6963219577357496, 2.7703523415996876, 4.14906465734074, 4.33577466528572]
num_type_to_param = {'17': 1.5615528128088303, 'M': 2.6963219577357496, 'P': 2.7703523415996876, 'M2': 4.14906465734074, 'P2': 4.33577466528572}
num_param_to_type = {1.5615528128088303: '17', 2.6963219577357496: 'M', 2.7703523415996876: 'P', 4.14906465734074: 'M2', 4.33577466528572: 'P2'}