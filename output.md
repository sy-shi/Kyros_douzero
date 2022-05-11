## Douzero自我对弈数据提取
by ssy
- 自我对弈命令：
  ``` cmd
  python my_data_generator.py --num_games 1000 --output true --output_path PATHNAME
- 输出文件名： **`output.pkl`**
- 读取方法：
``` python
import pickle
import numpy as np
np.set_printoptions(threshold = np.inf)
f = open('output.pkl','rb')
data = pickle.load(f)
```
- **`data`** 数据类型：list，每个元素储存一次完整对局的信息如下:
``` python
for info in data:
    print(info['init_hand_cards'])  #初始手牌情况 --> dict
    print(info['card_play_action_landlord']) #本局地主出牌情况 -->list
    print(info['card_play_action_landlord_down'])
    print(info['card_play_action_landlord_up'])
    print(info['landlord_win']) #本局地主胜负状况 --> int
    print(info['landlord_score']) #本局地主得分 --> double
    print(info['farmer_win'])
    print(info['farmer_score'])
```

