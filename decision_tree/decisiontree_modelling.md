# 建模
- **站在地主角度进行建模**

**输出** $Y$: **地主** 在 **当前局势** $X$ 下的获胜预测 | $Y \in \{-1,1\}$\
**输入** $X$: 对于地主的当前局势:
- idea1: X = initial state
- idea2: X = any current state 

映射: $\hat{Y} = f(T, X)$

---

## 特征空间 $X \in \mathcal{X}$ 建模
需要将当前牌局分解为不同的特征才能使用 **if-then** 决策树
### 1.只考虑initial state
- suppose:知道所有agent的手牌

$$
X_0 = (\text{地主炸弹值},\text{农民上炸弹值},\text{农民下炸弹值}\text{地主3张值},\text{农民上3张值},\text{农民下3张值}\text{...顺子值},\text{对子值},\text{单张值},\text{单张个数})
$$
- 3张，顺子为除去炸弹的结果；对子，单张为除去炸弹、三张、顺子的结果
- 实际特征 $X = \Alpha X_0$, $\Alpha$ 为根据大小确定的系数
```python
Card2Column = {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
               11: 9, 12: 10, 13: 11, 14: 12, 17: 13, 
               20: 14, 30: 15}
```
- 王炸 = 15
- 顺子的值 = $\Sigma$ (长度-4) $\times$ 中值
- 单张的值 = $\Sigma$大小
  
### 2.考虑state at any time
- suppose: unperfect information $\Rightarrow$ 不知道农民剩余手牌
- 可以用于决策如何出牌
$$
X = \left(\text{各自出牌轮数},|\text{各自剩余牌数},| \text{地主剩余炸弹值},\text{三张值},\text{顺子值},\text{对子值},\text{单张值},\text{单张个数},|\text{已出}\right)
$$
- “轮数”包括过牌，体现游戏进行的时序
- “已出”包含各agent已出炸弹值，顺子值...

---

## 模型 $Y_0=f(T,X): \mathcal{X} \mapsto \mathcal{Y}_0 \subset \mathbb{R}$
- ID3, C4.5为分类模型，构建多叉树，产生的分叉容易过多，泛化不好
- 利用 **`CART`** 构建二叉树回归模型
- 根据 $Y = \mathrm{sign}\left(f(T,X)\right)$由连续的$Y_0$映射到$\{-1,1\}$