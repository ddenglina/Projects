from GCForest import gcForest
import _pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import datetime
import pandas as pd
import scipy.io as sio
# import tensorflow as tf

starttime = datetime.datetime.now()

# file_list = ["s01"]
file_list=['s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s22', 's23','s24', 's25', 's21', 's26', 's27', 's28', 's29', 's30','s31','s32' ] #
test_accuracy_all_sub=np.zeros(shape=[0],dtype=float)
mean_accuracy_all=0

for data_file in file_list:
    # 读文件数据
    print('sub:', data_file) # ‘s01’
    cnn_suffix = ".mat_cnn_dataset.pkl"
    rnn_suffix = ".mat_rnn_dataset.pkl"
    label_suffix = ".mat_labels.pkl"
    arousal_or_valence = 'valence'
    dataset_dir = "./deap_shuffled_data/" + arousal_or_valence + "/"
    #load training set
    data_cnn = pickle.load(open(dataset_dir + data_file + cnn_suffix, 'rb'), encoding='utf-8')
    data_rnn = pickle.load(open(dataset_dir + data_file + rnn_suffix, 'rb'), encoding='utf-8')
    label = pickle.load(open(dataset_dir + data_file + label_suffix, 'rb'), encoding='utf-8')

    X_cnn = data_cnn  # (2400, 128, 9, 9)
    X_rnn = data_rnn  # (2400,128,32)
    y = label         # (2400,)

    fold = 10
    count = 0
    test_accuracy_all_fold = np.zeros(shape=[0], dtype=float)
    mean_accuracy = 0
    for curr_fold in range(fold):
        # 十折交叉验证 分数据
        fold_size = X_cnn.shape[0] // fold  # 2400//10=240
        indexes_list = [i for i in range(len(X_cnn))] # [0,2400]
        # indexes = np.array(indexes_list)

        split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
        split = np.array(split_list)  # [0,240],[240,240*2],....
        X_cnn_te = X_cnn[split]  # 测试集数据
        X_rnn_te = X_rnn[split]
        y_te = y[split]          # 测试集标签
        split = np.array(list(set(indexes_list) ^ set(split_list))) # 测试集剩下的就是训练集的
        X_cnn_tr = X_cnn[split]  # 训练集数据
        X_rnn_tr = X_rnn[split]
        y_tr = y[split] # 训练集标签 # (1200,)

        print("count",count)
        print("train_x shape:", X_cnn_tr.shape)  # (2160,128,9,9)
        print("train_x shape:", X_rnn_tr.shape)  # (2160,128,32)
        print("test_x shape:", X_cnn_te.shape)   # (240,128,9,9)
        print("test_x shape:", X_rnn_te.shape)   # (240,128,32)

        # train_sample = y_tr.shape[0] # 2160
        # test_sample = y_te.shape[0]  # 240

        X_cnn_tr = X_cnn_tr.transpose(1, 0, 2, 3) # (128,2160,9,9)
        X_cnn_te = X_cnn_te.transpose(1, 0, 2, 3) # (128,240,9,9)

        print("Slicing Images...")
        X_cnn_tr_ = X_cnn_tr[0] # 初始化：第一个128的数据 (2160, 9, 9)
        X_cnn_tr_ = X_cnn_tr_.reshape(X_cnn_tr.shape[1], 81) # (2160, 9, 9) reshape-->(2160,81)
        gcf = gcForest(shape_1X=[9, 9], window=6, tolerance=0.0, min_samples_mgs=20, min_samples_cascade=10)
        X_cnn_tr_mgs_ = gcf.mg_scanning(X_cnn_tr_, y_tr)   # scanning

        X_cnn_te_ = X_cnn_te[0] # (240, 9, 9)
        X_cnn_te_ = X_cnn_te_.reshape(X_cnn_te.shape[1], 81)  # (240, 9, 9) reshape-->(240,81)
        X_cnn_te_mgs_ = gcf.mg_scanning(X_cnn_te_)

        X_cnn_tr_mgs = X_cnn_tr_mgs_
        X_cnn_te_mgs = X_cnn_te_mgs_

        for i in range(1, X_cnn_tr.shape[0]):
            X_cnn_tr_ = X_cnn_tr[i]
            X_cnn_tr_ = X_cnn_tr_.reshape(X_cnn_tr.shape[1], 81)
            gcf = gcForest(shape_1X=[9, 9], window=6, tolerance=0.0, min_samples_mgs=20, min_samples_cascade=10)
            X_cnn_tr_mgs_ = gcf.mg_scanning(X_cnn_tr_, y_tr)   # (1200,16)                 # 模型根据y_tr进行训练
            X_cnn_tr_mgs = np.concatenate((X_cnn_tr_mgs, X_cnn_tr_mgs_), axis=1)     # (1200,32) # concatenate
            #print('X_tr_mgs_.shape:', X_tr_mgs_.shape)
            X_cnn_te_ = X_cnn_te[i]
            X_cnn_te_ = X_cnn_te_.reshape(X_cnn_te.shape[1], 81)
            #print('X_te_mgs_.shape:', X_te_mgs_.shape)
            X_cnn_te_mgs_ = gcf.mg_scanning(X_cnn_te_)                         # 回调最后训练的随机森林模型
            X_cnn_te_mgs = np.concatenate((X_cnn_te_mgs, X_cnn_te_mgs_), axis=1)  # (1200,32)
        print('X_cnn_tr_mgs.shape:', X_cnn_tr_mgs.shape) # (1200, 2048)
        print('X_cnn_te_mgs.shape:', X_cnn_te_mgs.shape) # (1200, 2048)

        X_rnn_tr = X_rnn_tr.transpose(1, 0, 2)  # (128,1200,32)
        X_rnn_te = X_rnn_te.transpose(1, 0, 2)  # (128,1200,32)

        print("Slicing Sequence...")
        X_rnn_tr_ = X_rnn_tr[0]  # 初始化：第一个(1200, 32)数据
        # X_cnn_tr_ = X_cnn_tr_.reshape(X_cnn_tr.shape[1], 81)  # (2160, 9, 9) reshape-->(2160,81)
        gcf = gcForest(shape_1X=32, window=23, tolerance=0.0, min_samples_mgs=20, min_samples_cascade=10)
        X_rnn_tr_mgs_ = gcf.mg_scanning(X_rnn_tr_, y_tr)  # (1200,24)# scanning

        X_rnn_te_ = X_rnn_te[0]  # (1200,32)
        # X_cnn_te_ = X_cnn_te_.reshape(X_cnn_te.shape[1], 81)  # (240, 9, 9) reshape-->(240,81)
        X_rnn_te_mgs_ = gcf.mg_scanning(X_rnn_te_)  # (1200,24)

        X_rnn_tr_mgs = X_rnn_tr_mgs_
        X_rnn_te_mgs = X_rnn_te_mgs_

        for i in range(1, X_rnn_tr.shape[0]):
            X_rnn_tr_ = X_rnn_tr[i]
            # X_rnn_tr_ = X_rnn_tr_.reshape(X_rnn_tr.shape[1], 81)
            gcf = gcForest(shape_1X=32, window=23, tolerance=0.0, min_samples_mgs=20, min_samples_cascade=10)
            X_rnn_tr_mgs_ = gcf.mg_scanning(X_rnn_tr_, y_tr)                    # 模型根据y_tr进行训练
            X_rnn_tr_mgs = np.concatenate((X_rnn_tr_mgs, X_rnn_tr_mgs_), axis=1)    # concatenate
            #print('X_tr_mgs_.shape:', X_tr_mgs_.shape)
            X_rnn_te_ = X_rnn_te[i]
            # X_rnn_te_ = X_rnn_te_.reshape(X_rnn_te.shape[1], 81)
            #print('X_te_mgs_.shape:', X_te_mgs_.shape)
            X_rnn_te_mgs_ = gcf.mg_scanning(X_rnn_te_)    #(1200,48)                     # 回调最后训练的随机森林模型
            X_rnn_te_mgs = np.concatenate((X_rnn_te_mgs, X_rnn_te_mgs_), axis=1)
        print('X_rnn_tr_mgs.shape:', X_rnn_tr_mgs.shape)  # (1200, 3072)
        print('X_rnn_te_mgs.shape:', X_rnn_te_mgs.shape)


        X_tr_mgs = np.concatenate((X_cnn_tr_mgs,X_rnn_tr_mgs), axis=1)
        print("X_tr_mgs",X_tr_mgs.shape)
        X_te_mgs = np.concatenate((X_cnn_te_mgs,X_rnn_te_mgs), axis=1)
        print("X_tr_mgs", X_te_mgs.shape)

        print("Training MGS Random Forests...")
        _ = gcf.cascade_forest(X_tr_mgs, y_tr)                        # 使用多粒度扫描的输出作为级联结构的输入，这里要注意
        pred_proba = np.mean(gcf.cascade_forest(X_te_mgs), axis=0)    # 级联结构不能直接返回预测，而是最后一层级联Level的结果
        preds = np.argmax(pred_proba, axis=1)                         # 因此，需要求平均并且取做大作为预测值

        # test_sample = y_te.shape[0]
        # print("test_x shape:", X_te.shape)
        # evaluating accuracy
        accuracy = accuracy_score(y_true=y_te, y_pred=preds)
        print('gcForest accuracy : {}'.format(accuracy))
        test_accuracy_all_fold = np.append(test_accuracy_all_fold, accuracy)
        mean_accuracy += accuracy
        count += 1

    print(mean_accuracy / fold)
    summary = pd.DataFrame({'fold': range(1, fold + 1), 'test_accuracy': test_accuracy_all_fold})
    writer = pd.ExcelWriter(
          "./result/DEAP/" + arousal_or_valence + "/" + data_file  + ".xlsx")
    summary.to_excel(writer, 'summary', index=False)
    writer.save()
#     mean_accuracy_all += mean_accuracy / fold
#     test_accuracy_all_sub = np.append(test_accuracy_all_sub, mean_accuracy / fold)
#
# print('mean_accuracy_all:', mean_accuracy_all / 32)
# result = pd.DataFrame({'sub': range(1, 33), 'test_accuracy': test_accuracy_all_sub})
# writer = pd.ExcelWriter("./result/DEAP/" + arousal_or_valence + ".xlsx")
# result.to_excel(writer, 'result', index=False)
# writer.save()

endtime = datetime.datetime.now()
print(endtime - starttime)