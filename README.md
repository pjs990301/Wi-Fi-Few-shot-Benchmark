# Wi-Fi-Few-shot-Benchmark
<br>

## Table of Contents
1. Introduction
2. Related Work
3. System Architecture
4. 

<br>

## 1. Introduction
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

## 2. Related Work
### Channel State Information
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

### Few-shot Learning & Meta-Learning
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

### Vision Transformer
Vision Transformer introduces a Transformer model in the vision field, demonstrating better performance in image classification tasks. 
This model is flexible in responding to the length of the input, which is advantageous for various image sizes and ratios, and simplifies the data preprocessing process. 
It also provides consistent performance regardless of the length of the input to maintain the same level of accuracy. 
Vision Transformers can be utilized for a variety of image classification tasks and perform better than traditional CNN-based models.

<div align="center">
    <img alt="img_2.png" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FI6CZv%2Fbtq4W1uStWT%2FBBBI8YYnbCgfO8rKeZTK31%2Fimg.png" width="700">
</div>

<br>

## 3. System Architecture
The figure shows the proposed meta-transformer. 
Our proposed system can be largely divided into four stages, from model learning to model evaluation, and proceeds in the order of separating learning data, model training, adding new class data, and model evaluation.

<div align="center">
    <img alt="Architecture.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/Architecture.jpg?raw=true" width="700">
</div>

<br>

### 3.1 Separating learning data (configuring support set and query set)
Each data point ($x_i$, $y_i$) consists of an input $x$ and its class label $y$. 
When you define the number of classes in an episode as $N_c$ and set the number of classes in a training dataset to $K$, random sampling creates a support set and query set consisting of $S_k$ and $Q_k$. 
At this point, the $RandomSample(S, N)$ function is used to select $N$ elements uniformly and randomly without redundancy in the set $S$. 
The support set($S_k$) is composed of  $RandomSample(D_{vk}, N_s)$, and the query set($Q_k$) is composed of $RandomSample(D_{vk} \setminus S_k, N_q)$. 
Here, $N_c$ means the number of support data per class, and $N_q$ means the number of query data per class.

### 3.2 Model Train 
#### 3.2.1 Transformer
Figure shows the process of embedding CSI data by dividing it by patch based on the structure of the vision transformer. 
In order to generate input data of the same structure, the transformer splits the CSI data of each support set into patch units and flattens each patch to change it into a vector value. 
Each vector is subjected to a linear operation and embedding. 
To remember the location of each patch, a Position embedding token is attached and the matrix is used as the encoder's input value. 
Attention is performed in the encoder, and as a result, the extracted feature vector is derived based on the relationship between the time patches given as input.
<div align="center">
    <img alt="Architecture.png" src="https://github.com/pjs990301/Wi-Fi-Few-shot-Benchmark/blob/main/fig/Transformer.png?raw=true" width="700">
</div>

### 3.2.2 Prototypical Network
Prototype calculation is performed with the feature vector extracted by Transformer encoding.    
$$
c_k = \frac{1}{|S_k|} \sum \text{Encoder}(x_i) \quad (x_i, y_i)
$$

