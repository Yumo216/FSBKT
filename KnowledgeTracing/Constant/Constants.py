

datasets = {
    'ASSIST': 'ASSIST',
    'Eedi': 'Eedi',
    'SLP-phy': 'SLP-phy',
    'term-phy': 'term-phy',
    'SLP': 'SLP-all'
}


question = {
    'ASSIST': 6474,  # *2=12948
    'Eedi': 882,   # *2=1764
    'SLP-phy': 1450,  # *2=2900
    'term-phy': 115,  # *2=230
    'SLP-all': 1447  # *2=2894
}

skill = {
    'ASSIST': 197,
    'Eedi': 85,
    # 'SLP-phy': 50,
    'term-phy': 37,
    'SLP-all': 191,
}

clients = {
    'ASSIST': 42,
    'Eedi': 30,
    # 'SLP-phy': 27,
    'term-phy': 31,
    'SLP-all': 31,
}

Device_Num = 3  # GPU编号
DATASET = datasets['SLP']  # 'ASSIST', 'Eedi', 'SLP'
QUES = question[DATASET]
SKILL = skill[DATASET]
CLIENTS = clients[DATASET]


# Dataloader
BATCH_SIZE = 64
MAX_STEP = 50
EMB_DIM = 128

# RNN
INPUT = QUES * 2
HIDDEN = 256
RNN_LAYERS = 1
OUTPUT = QUES



# Training
LR = 0.001  # 固定模型后再调
EPOCH = 200
K_fold = 5  # 五折交叉验证


