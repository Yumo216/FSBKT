import json
import math
import numpy as np
from KnowledgeTracing.Constant import Constants as C
from sklearn.model_selection import train_test_split, KFold


class DataReader():
    def __init__(self, data_path, maxstep, num_ques, rate=0.8):
        self.data_path = data_path
        self.maxstep = maxstep
        self.num_ques = num_ques
        self.rate = rate  # 用于划分训练集和测试集

    def getData(self):
        print(f"loading {C.DATASET}'s data...")
        AllData = [[], [], []]  # AllData[0]: 联邦训练用，每个学校一份, AllData[1]: 全局验证集, AllData[2]: 全局测试集
        school_data = {}
        # DIS = []
        with open(self.data_path, 'r') as file:
            data = json.load(file)  # 加载整个 JSON 数组
            for item in data:
                school_id = item['school_id']
                student_id = item['student_id']
                question_id = item['question_id']
                grade = item['answer']
                know = item['skill']

                # 初始化每个学校的列表
                if school_id not in school_data:
                    school_data[school_id] = []

                # 添加数据到对应学校
                school_data[school_id].append({
                    "skill": know,
                    "question_id": question_id,
                    "student_id": student_id,
                    "grade": grade
                })

            # 处理每个学校的数据
            for school_id, data in school_data.items():
                ques = []
                ans = []
                Data = []
                id = None
                for item in data:
                    if id is None:
                        id = item['student_id']
                    if id != item['student_id']:

                        # 当 student_id 改变时，处理之前学生的数据
                        mlen = len(ques)
                        slices = mlen // self.maxstep + (1 if mlen % self.maxstep > 0 else 0)
                        for i in range(slices):
                            batch_data = np.zeros(shape=[self.maxstep, 3])
                            if mlen > 0:
                                if mlen >= self.maxstep:
                                    steps = self.maxstep
                                else:
                                    steps = mlen
                                for j in range(steps):
                                    batch_data[j][0] = ques[i * self.maxstep + j]  # q
                                    batch_data[j][2] = ans[i * self.maxstep + j] + 1  # ans(1,2)
                                    if ans[i * self.maxstep + j] == 1:
                                        batch_data[j][1] = ques[i * self.maxstep + j]  # q+r
                                    else:
                                        batch_data[j][1] = ques[i * self.maxstep + j] + self.num_ques
                                mlen = mlen - self.maxstep
                            Data.append(batch_data.tolist())
                        ques = []
                        ans = []
                        id = item['student_id']

                    ques.append(int(item['question_id']))
                    ans.append(int(item['grade']))

                # 处理最后一个学生的数据
                mlen = len(ques)
                slices = mlen // self.maxstep + (1 if mlen % self.maxstep > 0 else 0)
                for i in range(slices):
                    batch_data = np.zeros(shape=[self.maxstep, 3])
                    if mlen > 0:
                        if mlen >= self.maxstep:
                            steps = self.maxstep
                        else:
                            steps = mlen
                        for j in range(steps):
                            batch_data[j][0] = ques[i * self.maxstep + j]
                            batch_data[j][2] = ans[i * self.maxstep + j] + 1
                            if ans[i * self.maxstep + j] == 1:
                                batch_data[j][1] = ques[i * self.maxstep + j]
                            else:
                                batch_data[j][1] = ques[i * self.maxstep + j] + self.num_ques
                        mlen = mlen - self.maxstep
                    Data.append(batch_data.tolist())

                # ====== 五折交叉验证替换原有划分 ======
                k = C.K_fold  # 折数
                fold_idx = getattr(self, 'fold_idx', 0)  # 当前折编号，需在外部赋值
                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                Data = np.array(Data)
                folds = list(kf.split(Data))
                train_idx, test_idx = folds[fold_idx]
                train_data = Data[train_idx]
                test_data_full = Data[test_idx]
                val_size = int(len(test_data_full) * 0.5)
                val_data = test_data_full[:val_size]
                test_data = test_data_full[val_size:]

                AllData[0].append(list(train_data))  # 每个学校的训练集
                AllData[1] += list(val_data)  # 全局验证集
                AllData[2] += list(test_data)  # 全局测试集




        return AllData[0], np.array(AllData[1]), np.array(AllData[2])
