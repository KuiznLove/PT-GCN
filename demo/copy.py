import json

def merge_json_files(fileName):
    # 读取train1.json
    # fileName = 'test'
    with open(fileName+'1.json', 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    # 读取train2.json
    with open(fileName+'.json', 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    # 确保两个文件的长度相同
    if len(data1) != len(data2):
        raise ValueError("两个JSON文件的记录数量不相同")
    
    # 将train1中的c字段复制到train2中对应的记录
    for i in range(len(data1)):
        if 'postag' in data1[i]:
            data2[i]['postag'] = data1[i]['postag']
        if 'head' in data1[i]:
            data2[i]['head'] = data1[i]['head']
        if 'deprel' in data1[i]:
            data2[i]['deprel'] = data1[i]['deprel']
    
    # 将更新后的数据写回train2.json
    with open(fileName+'.json', 'w', encoding='utf-8') as f2:
        json.dump(data2, f2, ensure_ascii=False, indent=None)
        print("成功将"+fileName+"1.json中的字段合并到"+fileName+".json中")

if __name__ == "__main__":
    try:
        # fileName = 'train'
        merge_json_files('train')
        merge_json_files('test')
        merge_json_files('dev')
    except Exception as e:
        print(f"发生错误: {str(e)}")