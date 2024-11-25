import csv
import numpy as np

def read_mnist_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # 将每行的最后一个元素作为标签，其余元素作为数据
            label = int(row[-1])
            image = np.array(row[:-1], dtype=np.float32)  
            if label == 0:
                label = np.array([1,0,0,0,0,0,0,0,0,0])
            elif label == 1:
                label = np.array([0,1,0,0,0,0,0,0,0,0])
            elif label == 2:
                label = np.array([0,0,1,0,0,0,0,0,0,0])
            elif label == 3:
                label = np.array([0,0,0,1,0,0,0,0,0,0])
            elif label == 4:
                label = np.array([0,0,0,0,1,0,0,0,0,0])
            elif label == 5:
                label = np.array([0,0,0,0,0,1,0,0,0,0])
            elif label == 6:
                label = np.array([0,0,0,0,0,0,1,0,0,0])
            elif label == 7:
                label = np.array([0,0,0,0,0,0,0,1,0,0])
            elif label == 8:
                label = np.array([0,0,0,0,0,0,0,0,1,0])
            elif label == 9:
                label = np.array([0,0,0,0,0,0,0,0,0,1])
            labels.append(label)
            data.append(image)
        
    return np.array(data), np.array(labels)

def get_hyperparam(cfg):
    print(" ")
    print("Train set path:", cfg.train_set_path)
    print("Test set path:", cfg.test_set_path)
    print("Units:", cfg.units)
    print("Batch size:", cfg.batch_size)
    print("Epoches:", cfg.epoches)
    print("Learning rate:", cfg.lr)
    print(" ")
    return cfg.train_set_path, cfg.test_set_path, cfg.units, cfg.batch_size, cfg.epoches, cfg.lr

if __name__ == '__main__':
    # 示例用法
    file_path = '../data/optdigits.tra'
    data, labels = read_mnist_data(file_path)
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    print("First image:\n", data[99])
    print("First label:", labels[99])




    import numpy as np
    a = np.array([1, 2, 3, 4,5, 6, 7, 8, 9, 10] )
    # 从 [1, 2, 3, 4] 中随机选择 3 个数字（允许重复）
    for i in range(3):
        result = np.random.choice(a, replace=False, size = 3)
        print(result)
        
