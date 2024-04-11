# 打开数据集文件
import string

with open('../datasets/ASTE-Data-V2-EMNLP2020/lap14/train_triplets.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 初始化计数器
count = 0
countline=0
cout30=0
# 遍历数据集中的每一行
for line in lines:
    sentence, _ = line.split("####")
    # 去除句子中的标点符号
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence_length=len(sentence.split())

    if sentence_length > 20:
        count += 1
    if sentence_length > 30:
        cout30+=1
    countline+=1
print("句子个数为:", countline)  # 906
print("句子长度大于20的个数为:", count)   # 263
print("句子长度大于30的个数为:", cout30)  # 86