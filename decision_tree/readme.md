## 从output.pkl生成$(X,Y)$
- read_data.py中导入了特征生成类 **`FeatureGenerator`**
``` python
from feature_generator import FeatureGenerator
```
- **`FeatureGenerator`** 对output.pkl中的数据进行操作，生成两种特征。
- 有以下主要成员:
``` python
class FeatureGenerator():
    self.feature1 = [] 
    """ feature1 ---> 16维, full information (5*3)"""
    self.feature2_list = []
    """ 
    feature2 ---> 27维, unperfect information
    每个agent已出牌轮次,每个agent剩余手牌数量,5*3,5*1
    """
    self.Y = self.score
    """in Output space"""
```
#### feature1 定义(可更改)
- 每一局开始时三个agent综合牌面情况
- 每一个agent牌面定义为$s_i \in \mathbb{R}^5$
$$
s_i = \left[\text{bomb值},\text{triple值},\text{seq值},\text{double值},\text{single值}\right],\quad i = 0,1,2
$$
- 则state为
$$
\begin{aligned}
& X_1 \in \mathbb{R}^{15} \\
&X_1 = \left[s_0 \quad s_1 \quad s_2 \right]
\end{aligned}
$$
- 返回的feature1为$(Y,X) \in \mathbb{R}^{16}$

#### feature2 定义(可更改)
- 任意牌局下对于地主的局势信息
$$
X_2 = \left[\begin{matrix}p_0 & n_0 & \tilde s _0 & p_1 & n_1 & \tilde s _1 & p_2 & n_2 & \tilde s _2 & s_0\end{matrix}\right] \in \mathbb{R}^{26}
$$
其中$p_i$表示打过的轮次，体现状态出现的时间性，$n_i$表示i剩余的手牌数，$\tilde s_i$表示i出过的牌中各类的数量，$s_0$为地主目前牌面
- feature2返回$(Y,X)\in \mathbb{R}^{27}$

#### 对于各类牌面值的定义(可更改)
- 对于各个牌面，定义其数值大小
```python
Card2Num = {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
               11: 9, 12: 10, 13: 11, 14: 12, 17: 13, 
               20: 14, 30: 15}
```
- $\text{bomb值} = \Sigma b_k$ 将agent持有炸弹的大小相加 $\rightarrow$  eg. 5炸的大小为3
- $\text{triple值} = \Sigma t_k$ 除去炸弹后，triple相加 $\rightarrow$ eg. 777大小为5
- $\text{seq值} = \Sigma (n-4)\cdot (max+min)$ 除去炸弹后，顺子从长到短**不重复**相加 $\rightarrow$ eg. 456789大小为$(6-4)\cdot(4+9) = 26$
- $\text{double值} = \Sigma d_k$ 除去顺子，trip，炸弹后对子大小相加
- $\text{single值} = \Sigma s_k$ 剩余单牌相加