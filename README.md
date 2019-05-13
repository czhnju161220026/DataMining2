

# 数据挖掘实验3

*姓名：崔子寒  学号：161220026 Email: cuizihan@nju.edu.cn*

## 一、提交说明

本次实验提交文件分为三个部分，包括训练测试结果、可执行jar包和完整工程。其中训练测试结果是调用weka的API，使用不同的分类方法，对所有数据进行十折交叉验证得到的结果。可执行jar包可以在命令行下执行java -jar来运行，复现实验结果。完整工程是完整的工程代码，使用maven作为构建工具。

## 二、实验过程

#### 1. 实验思路

本次实验提供的文件格式是arff文件，这是weka工具包使用的数据的文件格式，因此使用weka工具包提供的机器学习算法进行训练。本次实验需要比较：

+ J48决策树算法
+ 朴素贝叶斯(Naive Bayes) 算法
+ 支持向量机(SVM)算法
+ 神经网络(Neural Network)
+ K近邻(KNN)算法

以及这些算法的使用Bagging集成学习在给定的10个arff数据文件上的表现。weka提供了GUI界面，可以直接在界面上选中文件和所用的算法进行学习，不过对于10个文件和总共10个分类器来说，要进行100次操作，是比较低效的。所以编写程序，调用weka提供的API，自动的完成上述过程，并将结果输出至文件。

#### 2. 代码设计

##### 2.1 添加依赖

maven项目比较容易添加weka依赖，在pom.xml中添加如下的依赖：

``` xml
<!-- https://mvnrepository.com/artifact/nz.ac.waikato.cms.weka/weka-stable -->
    <dependency>
      <groupId>nz.ac.waikato.cms.weka</groupId>
      <artifactId>weka-stable</artifactId>
      <version>3.8.0</version>
    </dependency>
```

之后就可以调用weka的API进行机器学习。

##### 2.2 程序设计

weka通过**Instances**类来加载arff文件，**Instances**类有一个单参数构造器，需要传入一个java.io.Reader对象，t通过Reader，**Instances**类导入arff文件信息。

通过阅读weka源码可以发现，所有的基分类器都实现了**classifier**接口，这个接口的定义如下：

``` java
public interface Classifier {
    void buildClassifier(Instances var1) throws Exception;
    double classifyInstance(Instance var1) throws Exception;
    double[] distributionForInstance(Instance var1) throws Exception;
    Capabilities getCapabilities();
}
```

这个接口有两个方法*buildClassifier*和*classifyInstance*，从字面意思理解，分别用来训练和测试。

如果要进行集成学习，weka.classifier.meta下提供了一些集成学习算法。以**Bagging**为例,**Bagging**继承了抽象类**SingleClassifierEnhancer**,这个抽象类有下面这个方法：

``` java
public void setClassifier(Classifier newClassifier) {
        this.m_Classifier = newClassifier;
    }
```

所以我们创建了集成学习器后，调用*setClassifier*来设置基学习器，之后就可以进行训练了。

如果要进行十折交叉验证，可以使用weka.classifiers.Evaluation类，这个类的部分定义如下：

``` java
public class Evaluation implements Serializable, Summarizable, RevisionHandler {
   ...
    public void crossValidateModel(Classifier classifier, Instances data, int numFolds, Random random, Object...){...}
                                   
    public String toSummaryString() {...}

    public String toClassDetailsString(){...}
    ...
```

我们主要用到三个方法：*crossValidateModel*方法需要传入的参数有：一个Classifier接口、数据集、交叉验证的折数和随机数种子。通过设定numFolds = 10，就可以进行十折交叉验证。*toSummaryString*将返回交叉验证的总体结果，包括正确分类数目，准确度。*toClassDetailsString*返回更多的性能度量标准，包括TPR，FPR，Precision，Recall等指标。

因此，大致的思路如下：

1. 首先加载所有的arff文件，将它们名称存储在一个List &lt;String&gt; L1中。
2. 创建所有的分类器对象，将它们储存在一个List&lt;Classifier&gt; L2中。
3. 两层循环，先从L2中取出一个分类器，然后在L1中的所有数据上分别进行十折交叉验证，输出结果。

``` java
 for(MyClassifier classifier : classifiers) {
 	for(String file : files) {
 		MyEvaluation.tenCrossValidate(classifier,file);
 		...
 	}
 }
```



## 三、 测试结果

#### 1. 程序运行

程序运行后，会在控制台输出每个分类器训练测试所用的总时长，输出的格式如下：

![运行结果](https://github.com/czhnju161220026/image/blob/master/res.png?raw=true)

并将每个分类器的表现输入至result目录下的文件中：

![log](https://github.com/czhnju161220026/image/blob/master/log.png?raw=true)

#### 2 性能度量

经过统计，得到每一个分类器和对应的集成分类器在每个文件上的表现，得到下表：

<table style="text-align:center">
    <tr>
        <th rowspan='2' colspan='2'></th>
        <th colspan='2'> J48 </th>
        <th colspan='2'>贝叶斯</th>
        <th colspan='2'>SVM</th>
        <th colspan='2'>神经网络</th>
        <th colspan='2'>KNN</th>
    </tr>  
    <tr>
        <td>Acc(%)</td> <td>ROC</td> 
        <td>Acc(%)</td> <td>ROC</td> <td>Acc(%)</td> <td>ROC</td>
        <td>Acc(%)</td> <td>ROC</td> <td>Acc(%)</td> <td>ROC</td>
    </tr>
    <tr>
        <td rowspan='2'>breast-w</td>
        <td>普通</td>
        <td>94.56</td> <td>0.955</td>
        <td>95.99</td> <td>0.988</td>
        <td>97.00</td> <td>0.968</td>
        <td>95.28</td> <td>0.986</td>
        <td>95.14</td> <td>0.973</td>
    </tr>
    <tr>
        <td>集成</td>
        <td>96.28</td> <td>0.985</td>
        <td>95.85</td> <td>0.991</td>
        <td>97.00</td> <td>0.975</td>
        <td>95.99</td> <td>0.989</td>
        <td>95.85</td> <td>0.987</td>
    </tr>
<tr>
    <td rowspan='2'>colic</td>
    <td>普通</td>
    <td>85.33</td> <td>0.813</td>
    <td>77.99</td> <td>0.842</td>
    <td>82.61</td> <td>0.809</td>
    <td>80.43</td> <td>0.857</td>
    <td>81.25</td> <td>0.802</td>
</tr>
<tr>
    <td>集成</td>
    <td>85.59</td> <td>0.864</td>
    <td>77.99</td> <td>0.842</td>
    <td>83.97</td> <td>0.868</td>
    <td>84.51</td> <td>0.876</td>
    <td>81.25</td> <td>0.824</td>
</tr>
<tr>
    <td rowspan='2'>credit-a</td>
    <td>普通</td>
    <td>86.09</td> <td>0.887</td>
    <td>77.68</td> <td>0.896</td>
    <td>84.93</td> <td>0.856</td>
    <td>83.62</td> <td>0.895</td>
    <td>81.16</td> <td>0.808</td>
</tr>
<tr>
    <td>集成</td>
    <td>86.81</td> <td>0.928</td>
    <td>77.83</td> <td>0.896</td>
    <td>85.22</td> <td>0.888</td>
    <td>85.07</td> <td>0.908</td>
    <td>81.30</td> <td>0.886</td>
</tr>  
<tr>
    <td rowspan='2'>credit-g</td>
    <td>普通</td>
    <td>70.50</td> <td>0.639</td>
    <td>75.40</td> <td>0.787</td>
    <td>75.10</td> <td>0.671</td>
    <td>71.50</td> <td>0.730</td>
    <td>72.00</td> <td>0.660</td>
</tr>
<tr>
    <td>集成</td>
    <td>73.30</td> <td>0.753</td>
    <td>74.80</td> <td>0.787</td>
    <td>75.40</td> <td>0.754</td>
    <td>76.10</td> <td>0.776</td>
    <td>72.10</td> <td>0.694</td>
</tr>
<tr>
    <td rowspan='2'>diabetes</td>
    <td>普通</td>
    <td>73.83</td> <td>0.751</td>
    <td>76.30</td> <td>0.819</td>
    <td>77.34</td> <td>0.720</td>
    <td>75.39</td> <td>0.793</td>
    <td>70.18</td> <td>0.650</td>
</tr>
<tr>
    <td>集成</td>
    <td>74.60</td> <td>0.798</td>
    <td>76.56</td> <td>0.817</td>
    <td>77.47</td> <td>0.747</td>
    <td>76.82</td> <td>0.822</td>
    <td>71.09</td> <td>0.725</td>
</tr>
<tr>
    <td rowspan='2'>hapatitis</td>
    <td>普通</td>
    <td>83.87</td> <td>0.708</td>
    <td>84.52</td> <td>0.860</td>
    <td>85.16</td> <td>0.756</td>
    <td>80.00</td> <td>0.823</td>
    <td>80.64</td> <td>0.653</td>
</tr>
<tr>
    <td>集成</td>
    <td>83.87</td> <td>0.865</td>
    <td>85.81</td> <td>0.890</td>
    <td>85.81</td> <td>0.828</td>
    <td>84.52</td> <td>0.846</td>
    <td>81.29</td> <td>0.782</td>
</tr>
<tr>
    <td rowspan='2'>mozilla4</td>
    <td>普通</td>
    <td>94.80</td> <td>0.954</td>
    <td>68.64</td> <td>0.829</td>
    <td>83.21</td> <td>0.838</td>
    <td>91.19</td> <td>0.940</td>
    <td>88.99</td> <td>0.877</td>
    </tr>
<tr>
    <td>集成</td>
    <td>95.11</td> <td>0.976</td>
    <td>68.74</td> <td>0.830</td>
    <td>83.12</td> <td>0.849</td>
    <td>91.28</td> <td>0.945</td>
    <td>88.86</td> <td>0.928</td>
</tr>
<tr>
    <td rowspan='2'>pc1</td>
    <td>普通</td>
    <td>93.33</td> <td>0.668</td>
    <td>89.18</td> <td>0.650</td>
    <td>92.97</td> <td>0.500</td>
    <td>93.60</td> <td>0.723</td>
    <td>92.06</td> <td>0.74</td>
</tr>
<tr>
    <td>集成</td>
    <td>93.60</td> <td>0.855</td>
    <td>88.91</td> <td>0.628</td>
    <td>93.15</td> <td>0.512</td>
    <td>93.33</td> <td>0.833</td>
    <td>91.07</td> <td>0.793</td>
</tr>
<tr>
    <td rowspan='2'>pc5</td>
    <td>普通</td>
    <td>97.46</td> <td>0.817</td>
    <td>96.42</td> <td>0.830</td>
    <td>97.17</td> <td>0.541</td>
    <td>97.10</td> <td>0.941</td>
    <td>97.29</td> <td>0.932</td>
</tr>
<tr>
    <td>集成</td>
    <td>97.53</td> <td>0.959</td>
    <td>96.48</td> <td>0.842</td>
    <td>97.18</td> <td>0.572</td>
    <td>97.31</td> <td>0.954</td>
    <td>97.37</td> <td>0.953</td>
</tr>
<tr>
    <td rowspan='2'>waveform-5000</td>
    <td>普通</td>
    <td>75.08</td> <td>0.813</td>
    <td>80.00</td> <td>0.941</td>
    <td>86.68</td> <td>0.918</td>
    <td>83.56</td> <td>0.954</td>
    <td>73.62</td> <td>0.779</td>
</tr>
<tr>
    <td>集成</td>
    <td>81.20</td> <td>0.936</td>
    <td>79.98</td> <td>0.941</td>
    <td>86.26</td> <td>0.944</td>
    <td>85.68</td> <td>0.962</td>
    <td>74.46</td> <td>0.880</td>
    </tr>
</table>


​    

计算每个分类器在所有数据集上的平均准确率，以及它们的集成训练的平均准确率，得到下表:

<table>
    <tr>
        <th></th>
        <th > J48 </th>
        <th >贝叶斯</th>
        <th >SVM</th>
        <th >神经网络</th>
        <th >KNN</th>
    </tr>
    <tr>
        <td>平均ACC</td>
        <td>85.49</td>
        <td>82.21</td>
        <td>86.22</td>
        <td>85.17</td>
        <td>83.23</td>
    </tr>
    <tr>
        <td>平均ACC(集成)</td>
        <td>86.79</td>
        <td>82.29</td>
        <td>86.46</td>
        <td>87.06</td>
        <td>83.46</td>
    </tr>
    <tr>
        <td>平均提升</td>
        <td>1.30</td>
        <td>0.08</td>
        <td>0.24</td>
        <td>1.89</td>
        <td>0.23</td>
    </tr>
</table>

可视化数据：

![acc](https://github.com/czhnju161220026/image/blob/master/acc.png?raw=true)

#### 3. 时间代价

在十折交叉验证的过程中，记录了每一个分类器一级它们的集成的时间消耗，数据如下表：

<table>
    <tr>
        <th rowspan='2'></th>
        <th colspan='2'>J48</th>
        <th colspan='2'>贝叶斯</th>
        <th colspan='2'>SVM</th>
        <th colspan='2'>神经网络</th>
        <th colspan='2'>KNN</th>
     </tr>
     <tr>
         <th>普通</th><th>集成</th>
         <th>普通</th><th>集成</th>
         <th>普通</th><th>集成</th>
         <th>普通</th><th>集成</th>
         <th>普通</th><th>集成</th>
    </tr>
    <tr>
        <td>用时(分钟)</td>
        <td>0.2</td>
        <td>1.6</td>
        <td>0.02</td>
        <td>0.2</td>
        <td>1.15</td>
        <td>11.23</td>
        <td>24.83</td>
        <td>261.2</td>
        <td>1.1</td>
        <td>10.2</td>
    </tr>
</table>

可视化数据:

![time](https://github.com/czhnju161220026/image/blob/master/time.png?raw=true)

#### 4. 实验结论

综合行能度量和时间代价，可以得到如下的结论：

+ 复杂的分类器例如神经网络，SVM在面对不同的数据集时都有较好的性能，而简单的分类器如：贝叶斯，决策树，KNN在某些数据上表现较好，但是在某些数据集上表现不好。
+ 集成学习可以有效提高分类器的行能。
+ 集成学习应该遵循"好而不同"的原则，选择基分类器时既应该考虑基分类器的行能，也应该考虑多个基分类器会不会效果相似，从而无法达到集成学习的目的。从结果来看，使用决策树和神经网络作为基分类器效果较好，而一些简单的分类器如朴素贝叶斯，KNN则不会带来性能的显著提升。
+ 简单的学习器训练速度快，而复杂的学习器训练速度很慢。所以在可以接受性能略低的情况下，可以选择决策树这样的分类器。



## 四、 优化Bagging KNN

#### 1. 关于Bagging

Bagging是直接基于自助采样法的集成学习方法，通过有放回的随机采样，可以得到T个含m个样本的采样集，然后基于每个采样集训练出一个基学习器，再将这些基学习器进行结合。对于分类任务，采用简单投票法；对于回归任务，采用简单平均法。

从偏差-方差分解的角度看，Bagging关注降低方差，所以在决策树、神经网络等容易受样本扰动的学习器上的效果更为明显。

#### 2. 调优

Weka中的Bagging集成学习有如下的参数：

<table>
    <tr>
        <th>参数名</th><th>类型</th><th>描述</th>
    </tr>
    <tr>
        <td>bagSizePercent</td>
        <td>int</td>
        <td>Size of each bag, as a percentage of the training set size.</td>
    </tr>
    <tr>
        <td>calcOutOfBag</td>
        <td>boolean</td>
        <td>Whether the out-of-bag error is calculated.</td>
    </tr>
    <tr>
        <td>classifier</td>
        <td>Classifier</td>
        <td>The base classifier to be used.</td>
    </tr>
    <tr>
    <td>numIterations</td>
    <td>int</td>
    <td>The number of iterations to be performed.</td>
</tr>
    <tr>
    <td>seed</td>
    <td>int</td>
    <td>The random number seed to be used.</td>
</tr>
</table>

基学习器我们指定为KNN，可以供调整的参数有*bagSizePercent*，*calOutOfBag*，*numIterations*和*seed*. 其中seed仅用于随机抽样，和性能关系不大，所以探究*bagSizePercent*，*calOutOfBag*和*numIterations*这三个参数和性能的关系。

以colic.arff的数据为例，通过调整参数得到准确率随着参数变化的数据

<table>
    <tr>
        <th>calOutOfBag</th>
        <th rowspan='2'>bagSizePercent</th>
        <th colspan='4'>numIterations</th>
    </tr>
	<tr>
        <td rowspan='2'>True</td>
        <td>10</td> <td>15</td> <td>20</td> <td>25</td>
    </tr>
    <tr>
        <td>100</td>  
        <td>81.25%</td> <td>81.52%</td> <td>81.52%</td> <td>81.25%</td>
    </tr>
    <tr>
        <td rowspan='5'>False</td>
        <th>bagSizePercent</th>
        <td>10</td> <td>15</td> <td>20</td> <td>25</td>
    </tr>
    <tr>
        <td>70</td>
        <td>80.71%</td> <td>80.52%</td> <td>80.43%</td> <td>80.98%</td>
    </tr>
    <tr>
        <td>80</td>
        <td>80.98%</td> <td>80.98%</td> <td>80.71%</td> <td>81.25%</td>
    </tr>
    <tr>
        <td>90</td>
        <td>81.25%</td> <td>81.25%</td> <td>80.98%</td>  <td>80.71%</td>
    </tr>
    <tr>
        <td>100</td>
        <td>80.71%</td> <td>81.25%</td> <td>81.25%</td> <td>80.98%</td>
    </tr>
</table>



可以看出在使用包外误差进行辅助计算，并且适当调整包的个数是可以带来性能提升的。当使用包外误差，包的个数为15个时，达到了81.52%，比默认参数提高了0.6个百分点。接着我们固定Bagging的参数，对KNN的参数进行调整。

KNN的主要参数包括近邻的个数K，距离度量函数，通过调整这两个参数得到如下数据：

<table>
    <tr>
        <th rowspan='2'>距离度量函数</th><th colspan='5'>K</th>
    </tr>
    <tr>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
        <td>5</td>
    </tr>
    <tr>
        <td>欧式距离</td>
        <td>81.52%</td>
        <td>81.52%</td>
        <td>82.01%</td>
        <td>82.88%</td>
        <td>82.34%</td>
    </tr>
    <tr>
        <td>曼哈顿距离</td>
        <td>80.43%</td>
        <td>83.15%</td>
        <td>83.42%</td>
        <td>81.52%</td>
        <td>82.34%</td>
    </tr>
</table>

在尝试不同的K和距离度量函数后，发现使用曼哈顿距离，K=3时正确率最高，为83.42%，较默认参数提升了2.6个百分点。

因此，可以总结出提升Bagging KNN的一些方法：

* 使用包外误差进行辅助计算
* 调整Bag的个数，确定一个比较好的基分类器数目
* 多尝试一些度量函数和近邻个数K



## 五、 实验总结和参考



这次实验学到了关于Weka工具包的用法，锻炼了自己的编程能力和数据分析能力，获益匪浅。

参考：

* [在程序里使用weka](https://www.cnblogs.com/fengfenggirl/p/associate_weka.html)

