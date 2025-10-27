import json
import math
import numpy as np
from KnowledgeTracing.Constant import Constants as C
from sklearn.model_selection import train_test_split


class DataReader():
    def __init__(self, data_path, maxstep, num_ques, rate=0.8):
        self.data_path = data_path
        self.maxstep = maxstep
        self.num_ques = num_ques
        self.rate = rate  # 用于划分训练集和测试集

    def getData(self):
        print('loading data...')
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

                # 划分 train / val / test = 60 / 20 / 20
                total = len(Data)
                train_end = int(total * 0.6)
                val_end = int(total * 0.8)

                TRAIN = Data[:train_end]
                VAL = Data[train_end:val_end]
                TEST = Data[val_end:]

                AllData[0].append(TRAIN)  # 每个学校的训练集
                AllData[1] += VAL  # 全局验证集
                AllData[2] += TEST  # 全局测试集

        # print(f"Loaded {len(AllData[0])} schools.")
        # print(f"Total train sequences: {sum(len(s) for s in AllData[0])}")
        # print(f"Validation sequences: {len(AllData[1])}")
        # print(f"Test sequences: {len(AllData[2])}")


        return AllData[0], np.array(AllData[1]), np.array(AllData[2])




