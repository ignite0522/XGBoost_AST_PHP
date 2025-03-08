---
typora-copy-images-to: upload

---

# 树模型



## 一、前言

最近在天池比赛中打了一下练手赛，主题是《恶意代码检测》，主要是检测php代码

看了论坛上好多人评论XGBoost是AI比赛大杀器，凑巧这场比赛的第一名也是使用的是XGBoost结合ast抽象语法树来实现的

这么好用，那就干，开学！！



## 二、决策树

阐述：将一个集合中的数据，按照不同特征不断划分，形成一颗树；树的叶子节点表示样本所属的类别。如果是回归树，可以把叶子节点的值求平均作为样本的输出结果值。

说白了就是二叉树，每个节点会做一次判断，叶节点即为最终分类的结果



那怎么做判断呢，很简单

常用的方法有信息增益（ID3）、信息增益率（C4.5）、基尼系数（CART），计算方法分别为

信息增益（ID3）：即为熵

![img](https://p1-jj.byteimg.com/tos-cn-i-t2oaga2asx/gold-user-assets/2020/2/21/17066291b6b314f3~tplv-t2oaga2asx-jj-mark:3024:0:0:0:q75.png)



信息增益率（C4.5）：              信息增益比率



基尼系数（CART）：                 p*（1-p）



这里来解释一下信息增益（ID3）

```
举例来说，假设S是一个关于布尔概念的有14个样例的集合，它包括9个正例和5个反例（我们采用记号[9+，5-]来概括这样的数据样例），那么S相对于这个布尔样例的熵为：

Entropy（[9+，5-]）=-（9/14）log2（9/14）-（5/14）log2（5/14）=0.940。


So，根据上述这个公式，我们可以得到：

如果S的所有成员属于同一类，则Entropy(S)=0；
如果S的正反样例数量相等，则Entropy(S)=1；
如果S的正反样例数量不等，则熵介于0，1之间（如下图所示）
```

![img](https://s2.loli.net/2024/12/06/oF9YATnOzmNpyW4.jpg)

可以看到数据量越靠近中间的值，熵越大，不确定性越高





## 三、集成学习(Bagging vs Boosting)

所谓集成学习，即先使用多个弱分类器对数据进行预测（效果一般不好），然后采用某些策略将所有弱分类器得到的结果集成起来，做为最终的预测结果，通俗比喻就是“三个臭皮匠赛过诸葛亮”

### bagging

每个分类器见到不同的样本，样本权重相同，训练出不同效果的分类器，分类器相互独立，然后将这一系列的分类器组合起来使用

**类比一下就是物理学上，弱分类器看成是元器件，所有元器件串联**

###  boosting

每个分类器见到相同的样本，但每个分类器样本权重不同，并且分类器之间相互依赖。核心问题： 1） 如何确定样本权重 2） 如何组合各个分类器

**类比一下就是并联**

### bagging vs. boosting

- 训练样本 有放回抽样

- 是否能并行，分类器训练过程是否独立

- boosting 一般要比 bagging的效果好，

  

- Boosting流派，各分类器之间有依赖关系，必须串行，比如Adaboost、GBDT(Gradient Boosting Decision Tree)、Xgboost

- Bagging流派，各分类器之间没有依赖关系，可各自并行，比如随机森林（Random Forest）





## 四、Adaboost

这里就不详细讲解，他不是主角,贴上大佬的解释

```
AdaBoost，是英文"Adaptive Boosting"（自适应增强）的缩写，由Yoav Freund和Robert Schapire在1995年提出。它的自适应在于：前一个基本分类器分错的样本会得到加强，加权后的全体样本再次被用来训练下一个基本分类器。同时，在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数。

    具体说来，整个Adaboost 迭代算法就3步：

1.初始化训练数据的权值分布。如果有N个样本，则每一个训练样本最开始时都被赋予相同的权值：1/N。
2.训练弱分类器。具体训练过程中，如果某个样本点已经被准确地分类，那么在构造下一个训练集中，它的权值就被降低；相反，如果某个样本点没有被准确地分类，那么它的权值就得到提高。然后，权值更新过的样本集被用于训练下一个分类器，整个训练过程如此迭代地进行下去。
3.将各个训练得到的弱分类器组合成强分类器。各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，使其在最终的分类函数中起着较大的决定作用，而降低分类误差率大的弱分类器的权重，使其在最终的分类函数中起着较小的决定作用。换言之，误差率低的弱分类器在最终分类器中占的权重较大，否则较小。
```





## 五、GBoost

说到Xgboost，不得不先从GBDT(Gradient Boosting Decision Tree)说起。因为xgboost本质上还是一个GBDT，但是力争把速度和效率发挥到极致，所以叫X (Extreme) GBoosted

与AdaBoost不同，GBoost每一次的计算是都为了减少上一次的残差，进而在残差减少（负梯度）的方向上建立一个新的模型

还是不懂的话，扒了一张图

![img](https://s2.loli.net/2024/12/06/iDc1RondLwmzIfJ.png)

后面的树的输入是  总数据和前面树输出之差

看一个大佬举的例子就知道啦

```
它会在第一个弱分类器（或第一棵树中）随便用一个年龄比如20岁来拟合，然后发现误差有10岁；
接下来在第二棵树中，用6岁去拟合剩下的损失，发现差距还有4岁；
接着在第三棵树中用3岁拟合剩下的差距，发现差距只有1岁了；
最后在第四课树中用1岁拟合剩下的残差，完美。
```

还有个例子：

现在我们使用GBDT来做这件事，由于数据太少，我们限定叶子节点做多有两个，即每棵树都只有一个分枝，并且限定只学两棵树。

我们会得到如下图所示结果：


![img](https://s2.loli.net/2024/12/06/vrHs5yWi6njG4Tt.png)在第一棵树分枝和图1一样，由于A,B年龄较为相近，C,D年龄较为相近，他们被分为左右两拨，每拨用平均年龄作为预测值。

此时计算残差（残差的意思就是：A的实际值 - A的预测值 = A的残差），所以A的残差就是实际值14 - 预测值15 = 残差值-1。
注意，A的预测值是指前面所有树累加的和，这里前面只有一棵树所以直接是15，如果还有树则需要都累加起来作为A的预测值。
残差在数理统计中是指实际观察值与估计值（拟合值）之间的差。“残差”蕴含了有关模型基本假设的重要信息。如果回归模型正确的话， 我们可以将残差看作误差的观测值。

进而得到A,B,C,D的残差分别为-1,1，-1,1。

然后拿它们的残差-1、1、-1、1代替A B C D的原值，到第二棵树去学习，第二棵树只有两个值1和-1，直接分成两个节点，即A和C分在左边，B和D分在右边，经过计算（比如A，实际值-1 - 预测值-1 = 残差0，比如C，实际值-1 - 预测值-1 = 0），此时所有人的残差都是0。

残差值都为0，相当于第二棵树的预测值和它们的实际值相等，则只需把第二棵树的结论累加到第一棵树上就能得到真实年龄了，即每个人都得到了真实的预测值。

换句话说，现在A,B,C,D的预测值都和真实年龄一致了。Perfect！
A: 14岁高一学生，购物较少，经常问学长问题，预测年龄A = 15 – 1 = 14
B: 16岁高三学生，购物较少，经常被学弟问问题，预测年龄B = 15 + 1 = 16

C: 24岁应届毕业生，购物较多，经常问师兄问题，预测年龄C = 25 – 1 = 24
D: 26岁工作两年员工，购物较多，经常被师弟问问题，预测年龄D = 25 + 1 = 26

原文链接：https://blog.csdn.net/v_JULY_v/article/details/81410574





## 六、XGBoost

终于来到XGBoost了，由于不是数学科班出生，这里的数学原理就不做过多阐述，以免画蛇添足，之后会贴上大佬的链接，注意，要求数学原理要大致看懂，搞懂算法实现代码部分的底层逻辑

学疏才浅，本来想自己举个例子的，发现还是原作者的例子好用

引用自xgboost原作者陈天奇的讲义PPT中





举个例子，我们要预测一家人对电子游戏的喜好程度，考虑到年轻和年老相比，年轻更可能喜欢电子游戏，以及男性和女性相比，男性更喜欢电子游戏，故先根据年龄大小区分小孩和大人，然后再通过性别区分开是男是女，逐一给各人在电子游戏喜好程度上打分，如下图所示。

![img](https://s2.loli.net/2024/12/06/sdWL7vQpRjOwumG.png)





训练出了2棵树tree1和tree2，类似之前gbdt的原理，两棵树的结论累加起来便是最终的结论，所以小孩的预测分数就是两棵树中小孩所落到的结点的分数相加：2 + 0.9 = 2.9。爷爷的预测分数同理：-1 + （-0.9）= -1.9。具体如下图所示

![img](https://s2.loli.net/2024/12/06/ezJvY7fdZBRuKy2.png)

###  xgboost目标函数

![img](https://s2.loli.net/2024/12/06/lDxfJQmIP4SZu8K.png)

误差/损失函数鼓励我们的模型尽量去拟合训练数据，使得最后的模型会有比较少的 bias。而正则化项则鼓励更加简单的模型。因为当模型简单之后，有限数据拟合出来结果的随机性比较小，不容易过拟合，使得最后模型的预测更加稳定。

这两者是对抗关系，矛盾矛盾

XGBoost和gboost最大的区别就是是用了二阶导（由泰勒公式推算而来）

### 矛：误差函数

误差函数中的 ![img](https://i-blog.csdnimg.cn/blog_migrate/9ab5085abb701b71d8d695852337a726.gif) 表示第![img](https://i-blog.csdnimg.cn/blog_migrate/9ab5085abb701b71d8d695852337a726.gif)个样本，![img](https://i-blog.csdnimg.cn/blog_migrate/4db61abce6ac545620ba9d0b5eaf4d93.gif) (![img](https://i-blog.csdnimg.cn/blog_migrate/78c417cd88f74b55b6a5b1f4e46ffbd9.png) − ![y_{i}](https://i-blog.csdnimg.cn/blog_migrate/dcb6103ea2e92b04093685a2f0c74eec.gif)) 表示第 ![img](https://i-blog.csdnimg.cn/blog_migrate/9ab5085abb701b71d8d695852337a726.gif) 个样本的预测误差，我们的目标当然是误差越小越好。

类似之前GBDT的套路，xgboost也是需要将多棵树的得分累加得到最终的预测得分

![img](https://s2.loli.net/2024/12/06/yzkVQJKIvUGFwSx.png)



### 盾：正则化项

正则项是为了防止模型过拟合的

这里的正则依据有：	一个是树里面叶子节点的个数

这是因为叶子节点过多，就可以说明树分叉太多了，如果不加以限制的话会伸出巨大的枝叶，可以完全契合数据，导致过拟合，这是模型训练的大忌



## 七、**分裂节点**

很有意思的一个事是，我们从头到尾了解了xgboost如何优化、如何计算，但树到底长啥样，我们却一直没看到。很显然，一棵树的生成是由一个节点一分为二，然后不断分裂最终形成为整棵树。那么树怎么分裂的就成为了接下来我们要探讨的关键。

看了很多大佬的文章，都是直接用的贪心算法

从树深度0开始，每一节点都遍历所有的特征，比如年龄、性别等等，然后对于某个特征，**先按照该特征里的值进行排序，然后线性扫描该特征进而确定最好的分割点**，最后对所有特征进行分割后，我们选择所谓的增益Gain最高的那个特征

说人话就是指针走到每个点时把所有的可以出现的情况都遍历一遍找出最好的情况，之后往下走，重复这个过程，白话就是，鼠目寸光，只看到眼前的利益（肯定有更好的方法等你发现）

而Gain如何计算呢？

![img](https://s2.loli.net/2024/12/06/X9ypLNuxOzCRbeI.png)



举例：

比如总共五个人，按年龄排好序后，一开始我们总共有如下4种划分方法：

把第一个人和后面四个人划分开
把前两个人和后面三个人划分开
把前三个人和后面两个人划分开
把前面四个人和后面一个人划分开
接下来，把上面4种划分方法全都各自计算一下Gain，看哪种划分方法得到的Gain值最大则选取哪种划分方法，经过计算，发现把第2种划分方法“前面两个人和后面三个人划分开”得到的Gain值最大，意味着在一分为二这个第一层层面上这种划分方法是最合适的。





## 八、代码实现bagging

### Bagging策略

- 首先对训练数据集进行多次采样，保证每次得到的采样数据都是不同的
- 分别训练多个模型，例如树模型
- 预测时需得到所有模型结果再进行集成

![image-20241206160317336](https://s2.loli.net/2024/12/06/hI5yjlBqO2PWgT6.png)



对比一下bagging集成方法和传统方法

```py
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

X,y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                  n_estimators = 500,
                  max_samples = 100,
                  bootstrap = True,
                  n_jobs = -1,
                  random_state = 42
)
bag_clf.fit(X_train,y_train)
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test,y_pred)
print(accuracy_score(y_test,y_pred))

tree_clf = DecisionTreeClassifier(random_state = 42)
tree_clf.fit(X_train,y_train)
y_pred_tree = tree_clf.predict(X_test)
accuracy_score(y_test,y_pred_tree)
print(accuracy_score(y_test,y_pred_tree))
```

![截屏2024-12-06 16.21.08_副本](https://s2.loli.net/2024/12/06/VASrWYRb8U6t54z.png)

可以看到集成算法正确率更高



在写一个画图函数，更加直观看到

```py
#定义画图函数
def plot_decision_boundary(clf,X,y,axes=[-1.5,2.5,-1,1.5],alpha=0.5,contour =True):
    x1s=np.linspace(axes[0],axes[1],100)
    x2s=np.linspace(axes[2],axes[3],100)
    x1,x2 = np.meshgrid(x1s,x2s)
    X_new = np.c_[x1.ravel(),x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1,x2,y_pred,cmap = custom_cmap,alpha=0.3)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1,x2,y_pred,cmap = custom_cmap2,alpha=0.8)
    plt.plot(X[:,0][y==0],X[:,1][y==0],'yo',alpha = 0.6)
    plt.plot(X[:,0][y==0],X[:,1][y==1],'bs',alpha = 0.6)
    plt.axis(axes)
    plt.xlabel('x1')
    plt.xlabel('x2')
    
    
plt.figure(figsize = (12,5))
plt.subplot(121)
plot_decision_boundary(tree_clf,X,y)
plt.title('Decision Tree')
plt.subplot(122)
plot_decision_boundary(bag_clf,X,y)
plt.title('Decision Tree With Bagging')
plt.show()
```

![image-20241206162453734](https://s2.loli.net/2024/12/06/JQ1vwX27ihUjyTm.png)

可以看到集成算法的曲线更加的平滑



## 九、代码实现Adaboost

```py
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf,X,y,axes=[-1.5,2.5,-1,1.5],alpha=0.5,contour =True):
    x1s=np.linspace(axes[0],axes[1],100)
    x2s=np.linspace(axes[2],axes[3],100)
    x1,x2 = np.meshgrid(x1s,x2s)
    X_new = np.c_[x1.ravel(),x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1,x2,y_pred,cmap = custom_cmap,alpha=0.3)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1,x2,y_pred,cmap = custom_cmap2,alpha=0.8)
    plt.plot(X[:,0][y==0],X[:,1][y==0],'yo',alpha = 0.6)
    plt.plot(X[:,0][y==0],X[:,1][y==1],'bs',alpha = 0.6)
    plt.axis(axes)
    plt.xlabel('x1')
    plt.xlabel('x2')


X,y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
m = len(X_train)

plt.figure(figsize=(14,5))
for subplot,learning_rate in ((121,1),(122,0.5)):
    sample_weights = np.ones(m)
    plt.subplot(subplot)
    for i in range(5):
        svm_clf = SVC(kernel='rbf',C=0.05,random_state=42)
        svm_clf.fit(X_train,y_train,sample_weight = sample_weights)
        y_pred = svm_clf.predict(X_train)
        sample_weights[y_pred != y_train] *= (1+learning_rate)
        plot_decision_boundary(svm_clf,X,y,alpha=0.2)
        plt.title('learning_rate = {}'.format(learning_rate))
    if subplot == 121:
        plt.text(-0.7, -0.65, "1", fontsize=14)
        plt.text(-0.6, -0.10, "2", fontsize=14)
        plt.text(-0.5,  0.10, "3", fontsize=14)
        plt.text(-0.4,  0.55, "4", fontsize=14)
        plt.text(-0.3,  0.90, "5", fontsize=14)
plt.show()
```

简单讲一下上述代码实现的功能



1.以SVM分类器为例来演示AdaBoost的基本策略

2.用for循环分别判断了两个学习率1和0.5

3.更新错误分类样本的权重

```
sample_weights[y_pred != y_train] *= (1+learning_rate)
```

这行代码会增加那些预测错误的样本的权重，增加的比例是 `(1 + learning_rate)`，这样后续训练时错误分类的样本会对模型有更大的影响

运行结果：

![image-20241206163749323](https://s2.loli.net/2024/12/06/KWw85ml4G1cxVtB.png)

1->5是逐步优化的过程

可以看到曲线逐步考虑到了未分类的数据



接下来对比一下boosting集成算法和传统算法

```py
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf,X,y,axes=[-1.5,2.5,-1,1.5],alpha=0.5,contour =True):
    x1s=np.linspace(axes[0],axes[1],100)
    x2s=np.linspace(axes[2],axes[3],100)
    x1,x2 = np.meshgrid(x1s,x2s)
    X_new = np.c_[x1.ravel(),x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1,x2,y_pred,cmap = custom_cmap,alpha=0.3)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1,x2,y_pred,cmap = custom_cmap2,alpha=0.8)
    plt.plot(X[:,0][y==0],X[:,1][y==0],'yo',alpha = 0.6)
    plt.plot(X[:,0][y==0],X[:,1][y==1],'bs',alpha = 0.6)
    plt.axis(axes)
    plt.xlabel('x1')
    plt.xlabel('x2')


X,y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                   n_estimators = 200,
                   learning_rate = 0.5,
                   random_state = 42
)


ada_clf.fit(X_train,y_train)


tree_clf = DecisionTreeClassifier(random_state = 42)
tree_clf.fit(X_train,y_train)


plt.figure(figsize = (12,5))
plt.subplot(121)
plot_decision_boundary(tree_clf,X,y)
plt.title('Decision Tree')
plt.subplot(122)
plot_decision_boundary(ada_clf,X,y)
plt.title('Decision Tree With Boosting')
plt.show()
```

![image-20241206164647223](https://s2.loli.net/2024/12/06/RGmwMAJHS9xOikL.png)

这个没有bagging更明显，看看就行，对比一下





## 十、代码实现GBoost

```py
import numpy as np
np.random.seed(42)
X = np.random.rand(100,1) - 0.5
y = 3*X[:,0]**2 + 0.05*np.random.randn(100)

from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth = 2)
tree_reg1.fit(X,y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth = 2)
tree_reg2.fit(X,y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth = 2)
tree_reg3.fit(X,y3)

X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1,tree_reg2,tree_reg3))


import matplotlib.pyplot as plt
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

plt.figure(figsize=(11,11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Residuals and tree predictions", fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.show()
```

简述一下这段代码

随机生成一些数据，得到的分布类似于一元二次函数

利用回归决策函数

```py
tree_reg1 = DecisionTreeRegressor(max_depth = 2)
tree_reg1.fit(X,y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth = 2)
tree_reg2.fit(X,y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth = 2)
tree_reg3.fit(X,y3)
```

这段代码实现了不断去拟合残差（即上一个模型未拟合到的数据）



至于之后的代码就是画图啦，照搬大佬师傅的代码

![image-20241206170544535](https://s2.loli.net/2024/12/06/u6DrmHMoB2AgeX9.png)

可以看到右边的三个图，整体上曲线逐步拟合了数据

左边的图是在拟合残差，可以卡到绿色的线逐步趋近平的直线



XGBoost是建立在GBoost上的

区别在于XGBoost 中每棵树的构建不仅仅基于残差，还引入了 **正则化**，从而减少过拟合的风险。XGBoost 使用二阶梯度信息来提高模型的训练速度和性能。



## 十一、调参

咱先来看看XGBoost会受哪些参数影响，看看官方怎么说的吧

### **核心参数 (Booster Parameters)**

#### `learning_rate` (或 `eta`)：

- **作用**：控制每棵树的贡献大小。较小的学习率通常会提高准确性，但需要更多的树来拟合数据。
- **推荐值**：0.01 到 0.3 之间。常用值：0.01, 0.05, 0.1
- **调节技巧**：小的 `learning_rate` 值通常需要增加 `n_estimators`（树的数量）来弥补。

#### `n_estimators`：

- **作用**：树的数量，等于基学习器（树）的数量。
- **推荐值**：与 `learning_rate` 成反比，较低的学习率时，需要增加 `n_estimators`。
- **调节技巧**：增大树的数量有助于提高模型拟合能力，但容易过拟合，尤其是在 `learning_rate` 较大的情况下。

####  `max_depth`：

- **作用**：树的最大深度，控制树的复杂度。较大的 `max_depth` 可以让模型捕捉到更复杂的关系，但也容易过拟合。
- **推荐值**：3 到 10。
- **调节技巧**：增大 `max_depth` 会增加模型的复杂度，但也会增加训练时间和过拟合的风险。

####  `min_child_weight`：

- **作用**：控制子叶节点的最小样本权重和最小样本数。增加该值可以减少模型的复杂度，避免过拟合。
- **推荐值**：1 到 10。
- **调节技巧**：增加 `min_child_weight` 值通常有助于减少过拟合，但过大可能导致欠拟合。

#### `subsample`：

- **作用**：控制每棵树在训练时使用的样本比例，防止过拟合。
- **推荐值**：0.5 到 1.0。
- **调节技巧**：较低的 `subsample` 会增加随机性，减小过拟合，但如果设置得过低，可能导致欠拟合。

#### `colsample_bytree`：

- **作用**：控制每棵树在构建时使用的特征比例。与 `subsample` 相似，但作用在特征维度上。
- **推荐值**：0.5 到 1.0。
- **调节技巧**：较小的 `colsample_bytree` 会让每棵树训练时使用不同的特征，这样可以防止过拟合。

### **其他高级参数 (Advanced Parameters)**

#### `gamma`：

- **作用**：控制树的分裂，增大 `gamma` 会增加树的复杂度。较大的 `gamma` 值会使得树分裂更加保守，从而减少过拟合。
- **推荐值**：0 到 1。
- **调节技巧**：增大 `gamma` 会减少分裂的次数，适合避免过拟合。

#### `scale_pos_weight`：

- **作用**：控制正负样本的不平衡，在不平衡分类问题中可以通过调整该参数提高性能。
- **推荐值**：1 到负样本/正样本比例。

#### `max_delta_step`：

- **作用**：避免在类别不平衡的数据集中，梯度下降过程中模型参数的更新过大。
- **推荐值**：默认值 0，通常只有在类别严重不平衡时才使用。

### 网格搜索 (Grid Search)

看了这么多参数，那怎么才能找到合适最优的呢

这里介绍一个方法叫网格搜索 (Grid Search)

```
网格搜索是一种暴力穷举方法，通过遍历指定范围的超参数来找到最佳的超参数组合。常用的网格搜索方法有 GridSearchCV
```

还有**随机搜索 (Random Search)**：

随机搜索是从超参数空间中随机选择几个参数进行评估，适用于搜索空间非常大的情况。



废话不多说，直接上代码实战

我们这里选取的参数有

learning_rate学习率，max_depth最大树深，n_estimators树的数量

```py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBRegressor(objective='reg:squarederror')

# 网格搜索参数
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)




pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.width', None)        # 自动换行


results = pd.DataFrame(grid_search.cv_results_)
print(results[['param_learning_rate', 'param_max_depth', 'param_n_estimators', 'mean_test_score', 'std_test_score']])
print("Best parameters:", grid_search.best_params_)

```

简单解释一下

**`mean_test_score`** 是每个参数组合的平均交叉验证得分,绝对值越小越好

**`std_test_score`** 是每个参数组合的标准差，反映了不同数据折中的得分波动，数值越大，波动越大

注意我们这里使用的cv=3，是三折交叉验证

```
          param_learning_rate param_max_depth param_n_estimators  mean_test_score  std_test_score
0                 0.01               3                100        -0.114076        0.012829
1                 0.01               3                200        -0.087061        0.012155
2                 0.01               3                300        -0.078339        0.010817
3                 0.01               5                100        -0.096786        0.004698
4                 0.01               5                200        -0.078733        0.005907
5                 0.01               5                300        -0.076845        0.006330
6                 0.01               7                100        -0.102251        0.006768
7                 0.01               7                200        -0.090281        0.009028
8                 0.01               7                300        -0.092943        0.010753
9                 0.05               3                100        -0.075971        0.008423
10                0.05               3                200        -0.076350        0.008706
11                0.05               3                300        -0.079449        0.007189
12                0.05               5                100        -0.076551        0.004506
13                0.05               5                200        -0.086036        0.005398
14                0.05               5                300        -0.089266        0.005862
15                0.05               7                100        -0.096571        0.011469
16                0.05               7                200        -0.100736        0.011147
17                0.05               7                300        -0.101875        0.011291
18                 0.1               3                100        -0.076510        0.008103
19                 0.1               3                200        -0.082713        0.005937
20                 0.1               3                300        -0.086785        0.005418
21                 0.1               5                100        -0.085325        0.005892
22                 0.1               5                200        -0.089278        0.007298
23                 0.1               5                300        -0.090231        0.007570
24                 0.1               7                100        -0.100998        0.008897
25                 0.1               7                200        -0.102353        0.008431
26                 0.1               7                300        -0.102353        0.008431
Best parameters: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}

```



## 十二、实战之上万份php恶意代码文件检测

这个代码是打天池比赛看到的，魔改了一下

主要思路就是将php转换成AST抽象语法树，这样更利于在XGBoost之前的预处理阶段进行特征提取

之后把拿到的特征用封装好的XGBoost训练即可，最后的正确率达到97%多(我这里只取了1000份数据)



上代码

```python
from functools import partial
import pandas as pd
import plotly.express as px
import sklearn
import xgboost as xgb
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
pandarallel.initialize(progress_bar=True)
import json
import sys
from collections import Counter, defaultdict
from enum import Enum, auto
from pathlib import Path
import numpy as np
from scipy.stats import entropy
ZEND_AST_SPECIAL_SHIFT = 6
ZEND_AST_IS_LIST_SHIFT = 7
ZEND_AST_NUM_CHILDREN_SHIFT = 8
np.set_printoptions(threshold=np.inf)  # 设置显示所有元素


class PHPNodeKind(Enum):
    # special nodes
    ZEND_AST_ZVAL = 1 << ZEND_AST_SPECIAL_SHIFT
    ZEND_AST_CONSTANT = auto()
    ZEND_AST_ZNODE = auto()

    # declaration nodes
    ZEND_AST_FUNC_DECL = auto()
    ZEND_AST_CLOSURE = auto()
    ZEND_AST_METHOD = auto()
    ZEND_AST_CLASS = auto()
    ZEND_AST_ARROW_FUNC = auto()

    # list nodes
    ZEND_AST_ARG_LIST = 1 << ZEND_AST_IS_LIST_SHIFT
    ZEND_AST_ARRAY = auto()
    ZEND_AST_ENCAPS_LIST = auto()
    ZEND_AST_EXPR_LIST = auto()
    ZEND_AST_STMT_LIST = auto()
    ZEND_AST_IF = auto()
    ZEND_AST_SWITCH_LIST = auto()
    ZEND_AST_CATCH_LIST = auto()
    ZEND_AST_PARAM_LIST = auto()
    ZEND_AST_CLOSURE_USES = auto()
    ZEND_AST_PROP_DECL = auto()
    ZEND_AST_CONST_DECL = auto()
    ZEND_AST_CLASS_CONST_DECL = auto()
    ZEND_AST_NAME_LIST = auto()
    ZEND_AST_TRAIT_ADAPTATIONS = auto()
    ZEND_AST_USE = auto()

    # 0 child nodes
    ZEND_AST_MAGIC_CONST = 0 << ZEND_AST_NUM_CHILDREN_SHIFT
    ZEND_AST_TYPE = auto()
    ZEND_AST_CONSTANT_CLASS = auto()

    # 1 child node
    ZEND_AST_VAR = 1 << ZEND_AST_NUM_CHILDREN_SHIFT
    ZEND_AST_CONST = auto()
    ZEND_AST_UNPACK = auto()
    ZEND_AST_UNARY_PLUS = auto()
    ZEND_AST_UNARY_MINUS = auto()
    ZEND_AST_CAST = auto()
    ZEND_AST_EMPTY = auto()
    ZEND_AST_ISSET = auto()
    ZEND_AST_SILENCE = auto()
    ZEND_AST_SHELL_EXEC = auto()
    ZEND_AST_CLONE = auto()
    ZEND_AST_EXIT = auto()
    ZEND_AST_PRINT = auto()
    ZEND_AST_INCLUDE_OR_EVAL = auto()
    ZEND_AST_UNARY_OP = auto()
    ZEND_AST_PRE_INC = auto()
    ZEND_AST_PRE_DEC = auto()
    ZEND_AST_POST_INC = auto()
    ZEND_AST_POST_DEC = auto()
    ZEND_AST_YIELD_FROM = auto()
    ZEND_AST_CLASS_NAME = auto()

    ZEND_AST_GLOBAL = auto()
    ZEND_AST_UNSET = auto()
    ZEND_AST_RETURN = auto()
    ZEND_AST_LABEL = auto()
    ZEND_AST_REF = auto()
    ZEND_AST_HALT_COMPILER = auto()
    ZEND_AST_ECHO = auto()
    ZEND_AST_THROW = auto()
    ZEND_AST_GOTO = auto()
    ZEND_AST_BREAK = auto()
    ZEND_AST_CONTINUE = auto()

    # 2 child nodes
    ZEND_AST_DIM = 2 << ZEND_AST_NUM_CHILDREN_SHIFT
    ZEND_AST_PROP = auto()
    ZEND_AST_STATIC_PROP = auto()
    ZEND_AST_CALL = auto()
    ZEND_AST_CLASS_CONST = auto()
    ZEND_AST_ASSIGN = auto()
    ZEND_AST_ASSIGN_REF = auto()
    ZEND_AST_ASSIGN_OP = auto()
    ZEND_AST_BINARY_OP = auto()
    ZEND_AST_GREATER = auto()
    ZEND_AST_GREATER_EQUAL = auto()
    ZEND_AST_AND = auto()
    ZEND_AST_OR = auto()
    ZEND_AST_ARRAY_ELEM = auto()
    ZEND_AST_NEW = auto()
    ZEND_AST_INSTANCEOF = auto()
    ZEND_AST_YIELD = auto()
    ZEND_AST_COALESCE = auto()
    ZEND_AST_ASSIGN_COALESCE = auto()

    ZEND_AST_STATIC = auto()
    ZEND_AST_WHILE = auto()
    ZEND_AST_DO_WHILE = auto()
    ZEND_AST_IF_ELEM = auto()
    ZEND_AST_SWITCH = auto()
    ZEND_AST_SWITCH_CASE = auto()
    ZEND_AST_DECLARE = auto()
    ZEND_AST_USE_TRAIT = auto()
    ZEND_AST_TRAIT_PRECEDENCE = auto()
    ZEND_AST_METHOD_REFERENCE = auto()
    ZEND_AST_NAMESPACE = auto()
    ZEND_AST_USE_ELEM = auto()
    ZEND_AST_TRAIT_ALIAS = auto()
    ZEND_AST_GROUP_USE = auto()
    ZEND_AST_PROP_GROUP = auto()

    # 3 child nodes
    ZEND_AST_METHOD_CALL = 3 << ZEND_AST_NUM_CHILDREN_SHIFT
    ZEND_AST_STATIC_CALL = auto()
    ZEND_AST_CONDITIONAL = auto()

    ZEND_AST_TRY = auto()
    ZEND_AST_CATCH = auto()
    ZEND_AST_PARAM = auto()
    ZEND_AST_PROP_ELEM = auto()
    ZEND_AST_CONST_ELEM = auto()

    # 4 child nodes
    ZEND_AST_FOR = 4 << ZEND_AST_NUM_CHILDREN_SHIFT
    ZEND_AST_FOREACH = auto()

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def decode(cls, id):
        return cls(id).name
def str_entropy(s):
    """简单计算文本混淆度（熵）"""
    # 统计每个字符出现的次数
    counts = Counter(s)
    # 转换成概率分布
    probs = np.array(list(counts.values())) / len(s)
    # 计算熵
    ent = entropy(probs, base=2)
    return ent

def process_name(name):
    """处理节点名

    Example:
        "[ZVAL:\"c99shexit\"]" -> "ZVAL:c99shexit"
    """
    return name.replace("]", "").replace("[", "").replace('"', "")

# 风险函数
sinks = [
    "exec",
    "passthru",
    "shell_exec",
    "system",
    "ssh2_exec",
    "INCLUDE_OR_EVAL(eval)",
    "INCLUDE_OR_EVAL(include)",
    "INCLUDE_OR_EVAL(include_once)",
    "INCLUDE_OR_EVAL(require)",
    "INCLUDE_OR_EVAL(require_once)",
    "file_get_contents",
    "assert",
    "create_function",
    # ...
]

# 可控输入
sources = [
    "_GET",
    "_POST",
    "_COOKIE",
    "_REQUEST",
    "_FILES",
    "_SERVER",
    "_ENV",
    "_SESSION",
    # ...
]

def parse(content):
    """解析JSON文件，提取相关有效信息"""
    try:
        tree = json.loads(content)
    except Exception as err:
        print(f"Warning: {err}, {type(err)}")
        sys.exit(1)

    kind2cnt = defaultdict(int)
    lineno2nodecnt = defaultdict(int)
    sink2cnt = defaultdict(int)
    source2cnt = defaultdict(int)
    linenos = set()
    vals = []
    num_val_cutoff = 0
    num_val_tohex = 0

    def dfs(node):
        nonlocal kind2cnt
        nonlocal lineno2nodecnt
        nonlocal sink2cnt
        nonlocal source2cnt
        nonlocal linenos
        nonlocal vals
        nonlocal num_val_cutoff
        nonlocal num_val_tohex
        if "kind" in node and PHPNodeKind.has_value(node["kind"]):
            kind2cnt[PHPNodeKind.decode(node["kind"])] += 1
        if "lineno" in node:
            v = node["lineno"]
            linenos.add(v)
            lineno2nodecnt[v] += 1
        if "val" in node and isinstance(node["val"], str):
            v = node["val"]
            vals.append(v)
            for s in sources:
                source2cnt[s] += int(s == v)
        if "name" in node:
            v = process_name(node["name"])
            if ":" in v:
                sep_idx = v.index(":")
                vk, vv = v[0:sep_idx], v[sep_idx + 1 :]
                for s in sinks:
                    sink2cnt[s] += int(s == vk) + int(s == vv)
        if "val_cutoff" in node:
            num_val_cutoff += int(node["val_cutoff"])
        if "val_tohex" in node:
            num_val_tohex += int(node["val_tohex"])
        if "children" in node:
            [dfs(child) for child in node["children"]]

    dfs(tree)

    return (
        kind2cnt,
        lineno2nodecnt,
        sink2cnt,
        source2cnt,
        linenos,
        vals,
        num_val_cutoff,
        num_val_tohex,
    )

def process(p: Path):
    """特征抽取"""
    with p.open("r") as f:
        content = f.read()
    (
        kind2cnt,
        lineno2nodecnt,
        sink2cnt,
        source2cnt,
        linenos,
        vals,
        num_val_cutoff,
        num_val_tohex,
    ) = parse(content)

    f_kind2cnt = {k.name: 0 for k in PHPNodeKind}
    f_kind2cnt.update(kind2cnt)

    f_sink2cnt = {s: 0 for s in sinks}
    f_sink2cnt.update(sink2cnt)
    f_source2cnt = {s: 0 for s in sources}
    f_source2cnt.update(source2cnt)

    vals_entropy = [str_entropy(val) for val in vals]
    nodecnts = list(lineno2nodecnt.values())

    all_features = {
        "num_line": len(linenos),
        "max_lineno": max(linenos, default=0),
        "num_val_cutoff": num_val_cutoff,
        "num_val_tohex": num_val_tohex,
        "max_node_of_line": max(nodecnts, default=0), # max(单行节点数)
        "std_node_of_line": np.std(nodecnts) if len(nodecnts) > 0 else 0,
        "var_node_of_line": np.var(nodecnts) if len(nodecnts) > 0 else 0,
        "entropy": str_entropy("".join(vals)) if len(vals) > 0 else 0,
        "max_val_entropy": max(vals_entropy, default=0),
        "min_val_entropy": min(vals_entropy, default=0),
        "std_val_entropy": np.std(vals_entropy) if len(vals_entropy) > 0 else 0,
        "var_val_entropy": np.var(vals_entropy) if len(vals_entropy) > 0 else 0,
        "vals": vals,
    }

    all_features.update(f_kind2cnt) # 节点统计
    all_features.update(f_sink2cnt) # 危险函数统计
    all_features.update(f_source2cnt) # 输入源统计

    return all_features

import zipfile

dataset_folder = Path("/Users/guyuwei/PycharmProjects/PythonProject/deep_learning/webshell检测/xgb")


train_label_file = dataset_folder / "train.csv"
train_file_folder = dataset_folder / "train"
test_file_folder = dataset_folder / "test"

def fid_transform(file_id, train_or_test="train"):
    p = (train_file_folder if train_or_test == "train" else test_file_folder) / str(
        file_id
    )
    return process(p)

# metric
fbeta_score = partial(sklearn.metrics.fbeta_score, beta=0.5, average="binary")

# label mapping
id2label = {0: "white", 1: "black"}
label2id = {v: k for k, v in id2label.items()}
label_encode = lambda x: label2id[x]
label_decode = lambda x: id2label[x]

# 自定义的评估函数
def precision_recall_fbeta(preds, dtrain, threshold=0.5):
    labels = dtrain.get_label()
    preds = preds > threshold  # 把预测概率转换成二分类结果
    acc = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    fbeta = fbeta_score(labels, preds)
    return [("precision", acc), ("recall", recall), ("fbeta", fbeta)]

# read label file
df = pd.read_csv(train_label_file)
df = df.loc[df["type"] == "php"]
df.sample(n=1000)

y = df["label"].map(label_encode)
_X = df["file_id"].parallel_map(fid_transform)

X = pd.DataFrame(_X.copy().tolist())
X = X.drop("vals", axis=1)

# 检查结果
X[X.isna().any(axis=1)]

# 拆分训练集&验证集
Xtrain, Xeval, ytrain, yeval = train_test_split(
    X, y, test_size=0.1, random_state=1337
)
dtrain = xgb.DMatrix(data=Xtrain.values, feature_names=list(Xtrain.columns), label=ytrain)
deval = xgb.DMatrix(data=Xeval, label=yeval)
print(f"{deval.num_row()=}")

watchlist = [(deval, "eval"), (dtrain, "train")]


# 定义训练参数
param = {
    "max_depth": 20,
    "tree_method": "hist",
    "eta": 1,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "verbose": True,
    "early_stopping_rounds": 10,
}
num_round = 10

xgm = xgb.train(param, dtrain, num_round, evals=watchlist, feval=precision_recall_fbeta)
xgm.save_model('xgb_model.json')

fi = dict(
    sorted(xgm.get_score(importance_type="total_gain").items(), key=lambda x: x[1])
)
px.bar(y=list(fi.keys())[-20:], x=list(fi.values())[-20:])

e=xgm.predict(deval)
e_binary = (e > 0.5).astype(int)  # 将概率值转换为二分类标签
print("二分类结果：")
print(e_binary)

print("预测出的概率值：")
print(e)

fig = px.histogram(x=xgm.predict(deval), labels={'x':'probs', 'y':'count'})
fig.show()
```

其中PHP 节点类型定义[¶](https://tianchi.aliyun.com/mas-notebook/preview/470947/541168/-1?lang=#1.1-PHP-节点类型定义)

参考 PHP Zend: https://github.com/php/php-src/blob/PHP-7.4.15/Zend/zend_ast.h



###  刷分

#### 处理过拟合 (eval-fbeta:0.97654)

```py
param = {
    "max_depth": 20,
    "tree_method": "hist",
    "eta": 1,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "verbose": True,
    "early_stopping_rounds": 10,
}
num_round = 6
xgm = xgb.train(param, dtrain, num_round, evals=watchlist, feval=precision_recall_fbeta)
```

#### 提高分类阈值 (eval-fbeta:0.97798)

```py
param = {
    "max_depth": 20,
    "tree_method": "hist",
    "eta": 1,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "verbose": True,
    "early_stopping_rounds": 10
}
num_round = 7
xgm = xgb.train(param, dtrain, num_round, evals=watchlist, feval=partial(precision_recall_fbeta, threshold=0.98))
```

#### 处理类别不平衡问题 (eval-fbeta:0.98255)

```py
# 定义训练参数
neg_samples = ytrain.values.sum()
pos_samples = len(ytrain) - neg_samples
param = {
    "max_depth": 20,
    "tree_method": "hist",
    "eta": 1,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "verbose": True,
    "early_stopping_rounds": 10,
    # 增加黑样本权重
    "scale_pos_weight": 2 * neg_samples / pos_samples,
}
num_round = 7
xgm = xgb.train(param, dtrain, num_round, evals=watchlist, feval=partial(precision_recall_fbeta, threshold=0.98))
```

#### 增加特征 (eval-fbeta:0.98918)

使用TF-IDF 向量化器，通过 **TF-IDF** 特征工程将文本特征转换为数值特征，这个方法在datacorn比赛中使用自编码模型时，我也使用过，但效果并不好

```py
X = pd.DataFrame(_X.copy().tolist())
tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, max_features=200)
tfidf_train = tfidf.fit_transform(X["vals"].values.tolist()).todense().tolist()
tfidf_features = [
    # XGBoost 对特征名有格式要求
    "tfidf_" + f.replace("[", "").replace("]", "").replace("<", "")
    for f in tfidf.get_feature_names_out()
]

X = pd.concat([X, pd.DataFrame(tfidf_train, columns=tfidf_features)], axis=1)
X = X.drop("vals", axis=1)

Xtrain, Xeval, ytrain, yeval = train_test_split(
    X, y, test_size=0.1, random_state=1337
)

dtrain = xgb.DMatrix(data=Xtrain.values, feature_names=Xtrain.keys(), label=ytrain)
deval = xgb.DMatrix(data=Xeval, label=yeval)
watchlist = [(deval, "eval"), (dtrain, "train")]

# 定义训练参数
neg_samples = ytrain.values.sum()
pos_samples = len(ytrain) - neg_samples
param = {
    "max_depth": 20,
    "tree_method": "hist",
    "eta": 1,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "verbose": True,
    "early_stopping_rounds": 10,
    "scale_pos_weight": 2 * neg_samples / pos_samples,
}
num_round = 7
xgm = xgb.train(param, dtrain, num_round, evals=watchlist, feval=partial(precision_recall_fbeta, threshold=0.98))
```



## 总结

到此呢，对于树模型第一阶段就学到这里了，至于这段实战代码的解析还是放到之后的文章吧，毕竟要看懂他的预处理还是很有难度
