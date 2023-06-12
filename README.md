# Wi-Fi-Few-shot-Benchmark
This repository applies to HAR tasks using wi-fi sensing and tries various ways to resolve domain dependencies
<br>

## <b>Table of Contents</b>
1. Run Code
2. Introduction
3. Related Work
4. System Architecture
5. Experiments and evaluations
6. Conclusion

<br>

## <b>1. Run Code</b>
### <b>1.1 Requirements</b>
- python == 3.8
- pytorch
- torchvision
- numpy
- pandas

> You must install 'pytorch' and 'torchvision'
<pre><code>pip install pytorch torchvision</code></pre>

### <b>1.2 Dataset</b>
In the case of ReWiS datasets, please refer to the [link](https://github.com/niloobah/ReWiS), and the collected datasets are private,  
so if you need them, please email us at Pull request or p990301@gachon.ac.kr


### <b>1.3 Run</b>
If you want to do an experiment on all models and datasets
<pre><code> sh model.sh</code></pre>





## <b>2. Introduction</b>
Currently, activity recognition technology is applied to various services such as healthcare, smart home, and fitness. 
Although camera-based and wearable-based technologies have been mainly used in traditional methods, cameras have privacy leakage and limited range of filming problems, and wearables can cause additional costs and inconvenience. 
Recently, Wi-Fi Sensing, a detection technology using Wi-Fi, is attracting attention. 
Wi-Fi Sensing leverages indoor wireless networks to perform activity recognition, with fewer infrastructure needs and less privacy concerns compared to traditional methods.

Wi-Fi Sensing with deep learning model is based on advances in computer vision and natural language processing, and it is applied to various applications such as activity recognition, human authentication, and hand gesture recognition, showing good performance. 
However, learning a deep learning model requires a large amount of training data, and there are problems with insufficient data and difficulty predicting new classes. 
Data augmentation techniques can be used to address this, but there are drawbacks that require additional resources. 
Meta-learning can be introduced to enable training dataset shortages and activity awareness for new classes.

When choosing a deep learning model in Wi-Fi Sensing, performance and efficiency must be considered. 
CNNs have strengths in spatial pattern extraction, and RNNs are easy to process time series data, but have limitations. 
Recently, Transformer models have been actively utilized, achieving SOTAs in computer vision and natural language processing.

Therefore, we propose a meta-transformer that combines Transformer and meta-learning in Wi-Fi Sensing, which can understand the characteristics of time series data and improve generalization capabilities in various environments. 
This allows you to build an accurate and reliable Wi-Fi Sensing system.

<br>

## <b>3. Related Work</b>
### <b>3.1 Channel State Information</b>
The CSI is a radio received by the receiver Rx from the transmitter Tx
It is information on the detailed characteristics of the signal and the state of the channel.
At this time, the channel is measured at the subcarrier level,
This measurement is influenced by changes in the surrounding environment.
Wireless in the physical environment after diffraction, reflection, and scattering
Channel on the communication link reflecting how the signal is propagated be characteristic.

<div align="center">
    <h4>CSI Visualization </h4>
    <img alt="img_2.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/CSI.png?raw=true" width="900"/>
</div>

<br>

<div align="center">
    <h4>Domain dependency </h4>
    <img alt="img.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/img.png?raw=true" width="700"/>
</div>

### <b>3.2 Few-shot Learning & Meta-Learning</b>
Few-shot Learning is a machine learning technique that generalizes to only a small number of training data and classifies classes that you don't know before. 
To this end, we introduce a meta-learning approach to learn models for different tasks, and enable them to solve new tasks using a small number of samples. 
Meta-learning uses data that is divided into support set and query set, which is data that represents the domain of the job and is used for the model to learn. 
The support set allows the model to understand what features should be learned from the task, and use it to generalize about new tasks. 
Among the metric-based methods, Prototypical Networks can also predict classes that are not included in the learning data by calculating and classifying distances from class prototypes.
<div align="center">
    <img alt="img_2.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/img_2.png?raw=true" width="500"/>
    <img alt="img_3.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/img_3.png?raw=true" width="500"/>
</div>

<br>

### <b>3.3 Vision Transformer</b>
Vision Transformer introduces a Transformer model in the vision field, demonstrating better performance in image classification tasks. 
This model is flexible in responding to the length of the input, which is advantageous for various image sizes and ratios, and simplifies the data preprocessing process. 
It also provides consistent performance regardless of the length of the input to maintain the same level of accuracy. 
Vision Transformers can be utilized for a variety of image classification tasks and perform better than traditional CNN-based models.

<div align="center">
    <img alt="img_2.png" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FI6CZv%2Fbtq4W1uStWT%2FBBBI8YYnbCgfO8rKeZTK31%2Fimg.png" width="700">
</div>

<br>

## <b>4. System Architecture</b>
The figure shows the proposed meta-transformer. 
Our proposed system can be largely divided into four stages, from model learning to model evaluation, and proceeds in the order of separating learning data, model training, adding new class data, and model evaluation.

<div align="center">
    <img alt="Architecture.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/Architecture.jpg?raw=true" width="700">
</div>

<br>

### <b>4.1 Separating learning data (configuring support set and query set)</b>
Each data point ($x_i$, $y_i$) consists of an input $x$ and its class label $y$. 
When you define the number of classes in an episode as $N_c$ and set the number of classes in a training dataset to $K$, random sampling creates a support set and query set consisting of $S_k$ and $Q_k$. 
At this point, the $RandomSample(S, N)$ function is used to select $N$ elements uniformly and randomly without redundancy in the set $S$. 
The support set($S_k$) is composed of  $RandomSample(D_{vk}, N_s)$, and the query set($Q_k$) is composed of $RandomSample(D_{vk} \setminus S_k, N_q)$. 
Here, $N_c$ means the number of support data per class, and $N_q$ means the number of query data per class.

<br>

### <b>4.2 Model Train </b>
#### <b>4.2.1 Transformer</b>
Figure shows the process of embedding CSI data by dividing it by patch based on the structure of the vision transformer. 
In order to generate input data of the same structure, the transformer splits the CSI data of each support set into patch units and flattens each patch to change it into a vector value. 
Each vector is subjected to a linear operation and embedding. 
To remember the location of each patch, a Position embedding token is attached and the matrix is used as the encoder's input value. 
Attention is performed in the encoder, and as a result, the extracted feature vector is derived based on the relationship between the time patches given as input.
<div align="center">
    <img alt="Architecture.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/Transformer.png?raw=true" width="700">
</div>
<br>

### <b>4.2.2 Prototypical Network</b>
Prototype calculation is performed with the feature vector extracted by Transformer encoding.    
$$c_k = \frac{1}{\left\lvert S_k \right\rvert} \sum_{\quad (x_i, y_i)} \text{Encoder}(x_i) $$
As a result, the prototype forms a distribution representing each class in the embedding space. 
The prototype network determines a class for query points ($𝑥$) based on Softmax for the distance from the embedding space to the prototype.
$$p_{\emptyset}(y = k|x) = \frac{\exp\left(-d\left(\text{Encoder}(x), c_k\right)\right)}{\sum_{k'} \exp\left(-d\left(\text{Encoder}(x), c_{k'}\right)\right)}$$

<br>

### <b>4.3 Add New Class Data</b>
Meta-learning can quickly adapt to new tasks through learned training data, where classes of test data do not necessarily have to be included in the training data class. 
This is because meta-learning uses pre-meta information related to previously learned classes for classification of new classes. 
Determining the corresponding Unseen CSI data is also important because there are many types of Unseen CSI that are not included in the data class collected for training in actual Wi-Fi Sensing. 
Including Unseen CSI in existing training data, support set and query set are separated and tested.

<br>

### <b>4.4 Model evaluation</b>
Based on encoders learned from training data that do not include Unseen CSI, a prototype network is formed for the support set and query set containing Unseen. 
The model is evaluated by comparing the results of the corresponding prototype network with the labels of the real class.

<br>

## <b>5. Experiments and evaluations</b>
### <b>5.1 Dataset</b>
Two datasets were used for the dataset experiment, and the experiment was conducted based on the ReWiS dataset and the collected dataset. 
The contents of the data set can be found in Table
<br>
#### <b>5.1.1 ReWiS Dataset</b>
Nexmon CSI was used as a CSI extraction tool to use the Asus RTAC86U Wi-Fi router and collect CSI. 
The state of the 802.11ac channel at 5 GHz was measured at three locations using the Nexmon CSI tool. 
Among the 256 subcarriers of the 80 MHz channel, the guard and null subcarrier values are removed and 242x242 CSI data is provided through subcarrier compression. 
CSIs of activities such as walking, running, and writing were collected from various places (office, meeting room, and lecture room). 
This paper conducted an experiment on two of the three environments, an office and a classroom, and the number of antennas of Rx collected was set to four.
<br>
#### <b>5.1.2 Collection Datasets</b>
Using the Nexmon CSI tool, the state of the 802.11ac channel was measured at 5 GHz, and the state of the channel collected 64 subcarriers of 20 MHz. 
Unlike ReWiS, we constructed the dataset to resemble the real-world scenario using all 64 subcarriers collected.

<br>

<table align="center" style="margin-left: auto; margin-right: auto;">
  <tr>
    <th style="text-align:center;">Datasets </th>
    <th style="text-align:center;">ReWiS</th>
    <th style="text-align:center;">Collection Datasets</th>
  </tr>
  <tr>
    <td style="text-align:center;">CSI Tool</td>
    <td style="text-align:center;">Nexmon CSI</td>
    <td style="text-align:center;">Nexmon CSI</td>
  </tr>
  <tr>
    <td style="text-align:center;">Channel Information</td>
    <td style="text-align:center;">5GHz 80MHz</td>
    <td style="text-align:center;">5GHz 20MHz</td>
  </tr>
  <tr>
    <td style="text-align:center;">Activity number</td>
    <td style="text-align:center;">4</td>
    <td style="text-align:center;">5</td>
  </tr>
  <tr>
    <td style="text-align:center;">Activity names</td>
    <td style="text-align:center;">empty, jump, stand, walk</td>
    <td style="text-align:center;">empty, lie, sit, stand, walk</td>
  </tr>
  <tr>
    <td style="text-align:center;">CSI Size</td>
    <td style="text-align:center;">242 * 242</td>
    <td style="text-align:center;">64 * 64</td>
  </tr>
</table>

<br>

### <b>5.2 Experiment</b>
#### <b>5.2.1 Experiment 1: Meta Learning Necessity</b>
The above description describes experimental results that validate Wi-Fi sensing using supervised learning models such as CNN and RNN. 
Experiments used the ReWiS (Realistic Wireless Sensing) dataset, the training phase used the office dataset, and the testing phase used 20% of the office dataset to evaluate accuracy. 
Additionally, we used the classroom dataset to verify the generalization performance of the model. 
Experimental results show that for data in the same environment, all models performed accurate predictions for a given behavior. 
However, in other environments, the model did not perform predictions of behavior normally. 
In particular, the BiLSTM model showed the largest accuracy reduction rate, resulting in a performance degradation of approximately 60%. 
These results confirm that the supervised learning model learns the relationship between the input data and the class, so performance is significantly degraded when the environment changes.
Therefore, solving these problems may require other approaches, such as meta-learning.

<div align="center">
    <img alt="Architecture.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/need_meta.png?raw=true" width="500">
</div>

<br>

#### <b>5.2.2 Experiment 2: Applying Meta Learning</b>
Meta-learning allows models to learn different tasks and build on their learning experiences to quickly adapt to new tasks and gain generalization skills. 
Accordingly, the model was learned using office data from the ReWiS dataset, and in Experiment 1, the generalization performance of the model was verified using ReWiS's classroom data. 
In this experiment, we conducted an experiment by changing the size of the support set and query set considering the characteristics of meta-learning. 
The support set and query set sizes were set to 4 at the time of training, the training dataset and the test dataset had the same class, and the impact of the Unseen class was not considered. 
In this experiment, the performance was compared by applying CNN-based ProtoNet and Attention-based Vision Transformer (ViT).
As a result, the ViT model has about 23 times more parameters to learn than the ProtoNet model, takes a long time to infer, but has about 25 times less computation itself. 
Therefore, the ViT model may be advantageous for deployment in environments where weight reduction is required. 
Figure shows the results of ProtoNet and ViT models conducting learning with office datasets and testing on classroom datasets. 
Both models achieved 100% accuracy in training, and tests showed that the ViT model performed less well than the ProtoNet model. 
This demonstrates that applying ViT to meta-learning results in superior performance over ProtoNet and improves the generalization performance of the model using meta-learning.

<table align="center" style="margin-left: auto; margin-right: auto;">
  <tr>
    <th style="text-align:center;">Model</th>
    <th style="text-align:center;">Params (M)</th>
    <th style="text-align:center;">FLOPS (G)</th>
    <th style="text-align:center;">Elapsed time (s)</th>
  </tr>
  <tr>
    <td style="text-align:center;">ViT</td>
    <td style="text-align:center;">0.94</td>
    <td style="text-align:center;">0.011</td>
    <td style="text-align:center;">0.006</td>
  </tr>
  <tr>
    <td style="text-align:center;">ProtoNet</td>
    <td style="text-align:center;">0.04</td>
    <td style="text-align:center;">0.276</td>
      <td style="text-align:center;">0.001</td>
  </tr>
</table>
<br>
<div align="center">
    <img alt="Architecture.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/meta.png?raw=true" width="500">
</div>

    
#### <b>5.2.3 Experiment 3: Unseen Class</b>
Few-shot Learning is a method of learning models to have generalized classification capabilities for classes not included in the learning dataset. 
In this experiment, we conducted an experiment on Unseen classes that are not in the learning dataset to evaluate the generalization performance of Few-shot Learning. 
In the course of learning, the learning was conducted except for sit class data, and in the course of testing, the test was conducted including both the training dataset and sit class data.
We plotted a chaos matrix that visualized the experimental results. 
In the experiment, the training was conducted by setting the sizes of both the training support set and the training set to 5, and the sizes of the test support set and the test query set were set to 10 and 5, respectively.
The ViT model learned through Meta-Learning performed an accurate classification of Unseen CSIs not included in the learning.

<div align="center">
    <img alt="Architecture.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/confusion.png?raw=true" width="500">
</div>

<br>

## <b> 6. Conclusion</b>
It is important to have sufficient amount of data in learning the model. 
However, the collection of data can be costly and time-consuming. 
In the Wi-Fi CSI, the value of data included is changed according to the surrounding environment. 
As a result, we have the hassle of having to collect data every time for model learning. 
To this end, when introducing Meta-Learning, even if you have a small amount of data, you can quickly adapt to the environment and classify Unseen CSIs that are not included in the training dataset. 
In the case of CNN-based models, the size of each layer must be considered, and as the model goes deeper, the accuracy increases, but the amount of computation increases. 
Instead, Transformer is introduced and Meta Learning is applied to quickly generalize to a given environment and adapt to a new environment in the future.
