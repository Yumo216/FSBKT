import sys
sys.path.append('../')
import torch
import torch.utils.data as Data
from ..Constant import Constants as C
from ..data.readdataFL import DataReader
import numpy as np

def getDataLoader(batch_size, num_of_questions, max_step):
    path = getDatasetPaths(C.DATASET)
    train, vali, test = DataReader(path, C.MAX_STEP, C.QUES, rate=0.8).getData()

    # ---------- 统计 question_id 的最小值与最大值 ----------
    def extract_question_id_bounds(data):
        question_ids = []
        for seq in data:
            for step in seq:
                question_ids.append(int(step[0]))
        return min(question_ids), max(question_ids)

    train_flat = [step for school in train for step in school]
    train_min, train_max = extract_question_id_bounds(train_flat)
    val_min, val_max = extract_question_id_bounds(vali)
    test_min, test_max = extract_question_id_bounds(test)

    print(f"[Train] question_id range: [{train_min}, {train_max}]")
    print(f"[Vali] question_id range: [{val_min}, {val_max}]")
    print(f"[Test] question_id range: [{test_min}, {test_max}]")
    # -----------------------------------------------------

    trainLoader = []
    for idx, item in enumerate(train):
        # print(f"Index: {idx}, Type: {type(item)}, Length: {len(item)}")
        if len(item) == 0:
            print(f"Warning: dtrain at index {idx} is empty!")
        else:
            dtrain = torch.LongTensor(np.array(item).astype(float).tolist())
            trainLoader.append(Data.DataLoader(dtrain, batch_size=C.BATCH_SIZE, shuffle=False))  # 之前是shuffle=True
    dval = torch.LongTensor(vali.tolist())
    valiLoader = Data.DataLoader(dval, batch_size=C.BATCH_SIZE, shuffle=False)
    dtest = torch.LongTensor(test.tolist())
    testLoader = Data.DataLoader(dtest, batch_size=C.BATCH_SIZE, shuffle=False)


    return trainLoader, valiLoader, testLoader


def getDatasetPaths(dataset):
    if dataset == 'ASSIST':
        path = "../../KTDataset/ASSIST/ASSIST.json"

    elif dataset == 'Eedi':
        path = "../../KTDataset/Eedi/Eedi.json"

    else:
        raise ValueError(f"Dataset {dataset} is not recognized.")

    return path


