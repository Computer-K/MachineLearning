# coding:utf-8



import pandas as pd
import matplotlib.pyplot as plt
# 导入算法模块
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归算法模块
from sklearn.neighbors import KNeighborsClassifier  # 导入k近邻分类算法模块
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯算法模块
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类算法模块
from sklearn.svm import SVC  # 导入支持向量机分类模块
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类算法模块
from sklearn.neural_network import MLPClassifier  # 导入神经网络算法模块

from sklearn.model_selection import cross_val_score  # 导入交叉验证模块
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report


def main():
    # 第一步：数据导入
    names = ['variance', 'skewness', 'kurtosis', 'entropy', 'class']
    dataset = pd.read_csv('data_banknote_authentication.txt', delimiter=',', names=names)
    print('钞票鉴别数据集')
    print(dataset)  # [1372 rows x 5 columns]

    print('# 第二步：数据分析（查看数据分组 ）')
    print(dataset.groupby('class').size())

    # 第二步：数据分析（数据统计）
    print(dataset.describe())
    # (count,mean,std, min,0.25,0.5,0.75,max)

    # 第三步：数据可视化（单变量图--直方图）
    print('# 第三步：数据可视化（单变量图--直方图）')
    # 分别提取数据集中的特征变量和标签值
    data = dataset.iloc[range(0, 1372), range(0, 4)].values  # 提取训练数据
    target = dataset.iloc[range(0, 1372), range(4, 5)].values.reshape(1, 1372)[0]  # 提取标签值
    names_ = ['variance', 'skewness', 'kurtosis', 'entropy']
    # 绘制直方图
    plt.figure()  # 创建绘图对象
    for i, name in enumerate(names_):
        plt.subplot(2, 2, i + 1)
        plt.hist(data[:, i])  # 绘制直方图，仅包含数据，无标签
        plt.title(name)
    plt.tight_layout()  # 调整图形布局
    plt.show()
    # 第三步：数据可视化（单变量图--箱线图）
    plt.figure()  # 创建绘图对象
    for i, name in enumerate(names_):
        plt.subplot(2, 2, i + 1)
        plt.boxplot(data[:, i], whis=4)  # 绘制箱形图
        plt.title(name)
    plt.tight_layout()  # 调整图形布局
    plt.show()
    # 第三步：数据可视化（多变量图--散点图）
    plt.figure()  # 创建绘图对象
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i*4+j+1)
            plt.scatter(data[:, i], data[:, j])  # 绘制散点图
            plt.xlabel(names_[j])
            plt.ylabel(names_[i])
    plt.tight_layout()  # 调整图形布局
    plt.show()

    # 第四步：算法评估

    # 拆分数据集
    x, y = data, target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
    # train:test=0.8:0.2
    # 搭建模型
    models = []
    LRmodel = LogisticRegression(solver='liblinear')  # 搭建逻辑 回归模型
    kNNmodel = KNeighborsClassifier()  # 搭建k近邻 分类模型
    GNBmodel = GaussianNB()  # 搭建高斯朴素贝叶斯 分类模型
    DTreemodel = DecisionTreeClassifier(random_state=1)  # 搭建决策树 分类模型
    SVMmodel = SVC(gamma='auto', random_state=1)  # 搭建支持向量机 分类模型
    RFmodel = RandomForestClassifier(n_estimators=10, random_state=1)  # 搭建随机森林分类模型
    MLPmodel = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=1, max_iter=500)  # 搭建神经网络模型
    # 将各个模型加入models中
    models.append(('LRmodel', LRmodel))
    models.append(('kNNmodel', kNNmodel))
    models.append(('GNBmodel', GNBmodel))
    models.append(('DTreemodel', DTreemodel))
    models.append(('SVMmodel', SVMmodel))
    models.append(('RFmodel', RFmodel))
    models.append(('MLPmodel', MLPmodel))
    # 第四步：算法评估
    # 使用交叉验证法约束模型的训练过程，并估计每个模型的预测准确率

    for name, model in models:
        kfold = KFold(n_splits=10, random_state=100, shuffle=True)  # 10折交叉验证
        cv_scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        print('%s的预测准确率为：%f' % (name, cv_scores.mean()))
    # 第五步：训练与评估模型（支持向量机模型）

    SVMmodel = SVC(gamma='auto', random_state=1)
    SVMmodel.fit(x_train, y_train)
    # 对模型进行评估，并输出评估报告
    pred = SVMmodel.predict(x_test)
    re = classification_report(y_test, pred)
    print('支持向量机模型评估报告：')
    print(re)
    # 第五步：训练与评估模型（神经网络模型）
    MLPmodel = MLPClassifier(hidden_layer_sizes=(5, 5), random_state=1, max_iter=500)
    MLPmodel.fit(x_train, y_train)
    # 对模型进行评估，并输出评估报告
    pred = MLPmodel.predict(x_test)
    re = classification_report(y_test, pred)
    print('神经网络模型评估报告：')
    print(re)
    # 第六步：新数据预测（支持向量机模型）
    x_new = [[3.8216, 5.6661, -2.7074, -0.46611]]
    # 支持向量机模型预测新数据
    SVMscore = SVMmodel.predict(x_new)
    if SVMscore == 0:
        print("支持向量机模型预测结果：该钞票是假钞")
    else:
        print("支持向量机模型预测结果：该钞票是真钞")
    # 神经网络模型预测新数据
    MLPscore = MLPmodel.predict(x_new)
    if MLPscore == 0:
        print("神经网络模型预测结果：该钞票是假钞")
    else:
        print("神经网络模型预测结果：该钞票是真钞")


if __name__ == '__main__':
    main()
