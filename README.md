[toc]

## Run cmd:

- CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix_ricm --env-config=ue_cluster with
  env_args.map_name=ue_cluster
- CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix_ricm_transfer --env-config=ue_cluster with
  env_args.map_name=ue_cluster
- CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix_ricm_transfer --env-config=ue_cluster with
  env_args.map_name=ue_cluster lr=1e-5
- CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix_ricm_transfer --env-config=ue_cluster with
  env_args.map_name=ue_cluster lr=1e-5 mixer=vdn
- CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix_ricm_transfer --env-config=ue_cluster with
  env_args.map_name=ue_cluster lr=1e-5 mixer=vdn td_lambda=0

## 以 root 身份在指定的虚拟环境中运行python

http://www.siyuanblog.com/?p=31126
conda info --envs
export PATH=/home/ustc-lc1/miniconda3/bin:$PATH
export PATH=/home/ustc-5/.conda/bin:$PATH
临时改变，只能在当前的终端窗口中有效，当前窗口关闭后就会恢复原有的path配置:echo $PATH
source activate pymarl
sudo -E CUDA_VISIBLE_DEVICES=1 python3 src/main.py ...
-E 选项会保留当前用户的环境变量，包括已激活的虚拟环境。

## Finished:

### 1.16

- 完成 env_1 与多智能体控制算法的融合
- env_1 更新环境相关信息记录功能

### 1.17

- 降低 悬停和数据传输 的能耗参数
- 提高非法动作惩罚值（-5）
- 提高UAV覆盖UE奖励（取消归一化操作）
- 修改 env_1 每个时隙长度为 5
- 训练过程中打印整个测试episode的平均reward
- 调整 env_1 的 get_state 功能，返回：所有UE的位置（2维）、所有UAV的位置（3维）
- 调整了地图的大小（50*50）
- 现在可以通过配置文件设置 cluster 中心的位置和无人机的初始位置

### 1.18

- 将agent的action也记录下来
- 实现record
    - 系统
        - 每个时隙的UE覆盖率，公平指数，奖励值（整体），覆盖成功的UE cluster数、各个cluster在每个时隙的覆盖状态（成功或失败）
    - 无人机：
        - 每个时隙的动作（l,\theta,h）、能耗、服务用户数、位置、覆盖半径、动作是否非法、奖励值（个体）
    - UE：
        - 位置（启动时存一次）；每个时隙覆盖状态；至当前时隙被覆盖次数
-

sub_reward_total:[所有UE至当前时隙的覆盖率之和，当前实习总覆盖UE数/10，当前时隙cluster被成功覆盖的数量，无人机总能耗（除以了单无人机理论最大能耗），总惩罚项]

- 奖励设计更新
    - $$公平指数 \times \frac{至当前时隙所有\text{UE}覆盖率之和 + 当前时隙被成功覆盖\text{cluster}数量 +
      当前时隙总覆盖\text{UE}数/10}{系统所有无人机能耗} + 所有非法动作惩罚项$$
- 设计无人机初始位置与各UE cluster初始位置
- 设置各UE cluster半径固定为5
- 目前，reward_total理论最大为：1*(15+6+)/(6*0.55) =

### 1.19

- 环境中能量的计算优化

### 1.21

1. 对每个智能体基于本地reward进行训练
    - 算法配置文件中 learner: two_learner

2. 设置agent神经网络
    - 将无人机观测的信息全连接，将向量类信息单拎出来，提高其影响权重
    - 设置transferrable agent
    - 算法配置文件中 agent: "transfer"

3. 额外设置一个由高reward_total的data构成的经验缓冲区，设置不同比例的采样方案
    - 默认配置文件中 run: "two_sample"; is_two_sample: True; highreward_batch_size: 16

### 1.22

1. 原始观测修改：覆盖用户->服务用户，增加了 novelty 的更新
2. 动作空间设计修改，维度 72 -> 27
3. 状态设计修改：cluster 中心位置，无人机位置，障碍物位置、大小
4. 环境
    - 检查无人机的原始观测
    - env_1 检查正确性
    - 画图待补充：UE被覆盖时和所覆盖无人机的颜色一致
    -
   局部观测的remain_energy会成负值，一个思考：应该用energy_slot_cost代替remain_energy，因为现有能量并不能反应什么信息，并没有能量很低就要去充电这样的操作，所以存储单时隙消耗能量更有价值，能反应某动作对能耗的考虑，对于一个固定动作，能够反应其能耗共性。

### record 信息统计

- 系统：
  每个时隙的UE覆盖率 | 公平指数 | 奖励值（整体） | 覆盖成功的UE cluster数 | 各个cluster在每个时隙的覆盖状态（成功或失败）


- 无人机：
  每个时隙的动作（l,\theta,h） | 能耗 | 服务用户数 | 位置 | 覆盖半径 | 动作是否合法 | 奖励值（个体）

- UE：
  位置（启动时存一次） | 每个时隙覆盖状态 | 至当前时隙被覆盖次数

#### 智能体神经网络设计

1. 智能体输入构成：image部分+vector部分+last slot action+agent_id+global state
2. 对image部分，进行卷积操作（降维），然后与vector部分、last slot action、agent_id、global state进行cat
3. 将cat后的内容传入全连接层得到x
3. 由 state 构建超参数网络，得到 W 与 b，计算 W*x+b 得到输出作为 Q 值

`注：第3步的全连接层，如果效果不好，可用GRU来代替，测试效果`

## Questions:

- 奖励设计的一个不合理现象
    - UAV处于环境边缘 [99, 14, 3], 覆盖到了UE cluster，此时产生非法动作，但是奖励值是正的（覆盖到的UE数的正向奖励抵消了非法动作的惩罚项）
    - 解决思路
        - 合理搭配非法动作的惩罚项：
            - 确保只要有惩罚项，那么奖励就是负的
            - 当产生非法动作的时候，奖励就是惩罚项，没有正向奖励
        - 合理安排UE cluster的位置
            - 设置比较好的UE cluster位置初始化设定
            - *UAV初始化位置调整在各UE cluster附近*
- UAV的高度与覆盖（观测）半径的关系：
    - 要体现出无人机最优高度的最大覆盖程度 [1, 1, 2, 2, 5, 2, 2, 1, 1]
- 奖励函数的一个设计思路：
    - 将能耗放在分母里，只有非法动作才会引入惩罚项
- 一个比较关键的思路：
    - 对于QMIX应用在环境中，一个尴尬的现象是系统无法合理分辨出每个智能体局部的收益，导致有些无人机飞的很离谱。
    - 一个应该可行的思路是：不仅仅整个系统更新，还要各个无人机自己本地更新，把 reward_agents 利用起来。
    - **如何合理衡量（考量、度量）每个智能体动作给系统整体带来的收益是多智能体算法的关键**

### 1.19

- 稀疏奖励的现象很明显
    - 扩大UE cluster的范围
    - 再优化UE分布的设计
    - *内在好奇心*
- 一个对训练数据-经验池优化的想法
    - 一个可能存在的问题：经验池一共存储5000批，每次产生8批，从开始到末尾覆盖结束后，再从开始到末尾覆盖。那么，起始时刻的随机动作产生的数据会被覆盖，会不会导致经验池数据被神经网络主导的经验数据替代？导致训练时没有好的数据。
    - 将奖励值高的数据存下来，单独存一个经验池，然后分配采样的比例，每次都分别从两个经验池中采取两批数据
    - 高奖励值-高优先级采样
- **给每个无人机都配置一个智能体，然后将reward_total的训练和各个智能体本地reward的训练结合起来**

- actions的大小也可以降低：(l, $\theta$, h)中，l为0时，所有的角度选择都没有意义，环境里面需要修改一下，此外，将水平飞行动作(0,
  a ,2a)的设计取消了，(0, a)就可以。这样，动作空间维度应该是 3+3*8 = 27。
- 全局状态设计中，可以将所有UE的位置这一项去掉，用各cluster的中心位置来代替，这样状态空间维度就是：$6\times2+6\times3 + x=
  30+x$（x是障碍物位置，后面加入）

- 对学习器设计的一个考虑
    - 目前学习器更新时，只会由reward_total计算TD loss，然后backword实现对mixing network与agent network的更新
    - 优化方式：计算各agent的本地TD loss，然后backword更新agent network

*

*
对于通信网络领域中多智能体算法应用的一个痛点：很难应对有多种因素影响的优化目标（多目标优化问题），因为一个高的奖励会受到多类因素的影响，多智能体算法下，一方面，难评估各个智能体对系统收益的影响，另一方面，难评估每个动作导致的各项因素对收益的影响。
**

### 1.22

- *还有一个是，在原始observation的vector-class中是否可以将动作（上一时隙）是否合法也放进去？*

### 1.23

- 一个比较大的问题，设计了 transferrable agent 之后，误差特别大，过亿了。
    - 目前找到一个原因是超参数网络输出的参数都是正的并且太大，已对超参数网络添加 sigmoid 激活函数来降低数值
    - 还有一个原因是 lr 太大，0.01，这会导致网络更新过度，降不下去
    - 现在有两个超参数网络，这之间的影响太大了，神经网络训练无法收敛
- 一种设计奖励的思路
    - 类似于“追捕”的思想，无人机对于一个cluster的覆盖：
        - 当对该cluster的服务程度超过一定阈值时，则该智能体将该cluster视为“猎物”，奖励值为自己到cluster中心距离的“倒数”
        - 当对该cluster的覆盖程度高，但服务程度低，则不将该cluster视为“猎物”
- 还有一个隐藏的问题：
    - 现在环境没有终止状态，但智能体计算TD误差的时候会从末尾时隙往前迭代产生TD误差值。这样的话，强制让一个episode跑满episode_limit再结束是否不合情理？
    - 仔细研究一下 TD($\lambda$)

### 1.25

reward_total 重新设计
$$
公平指数 + \frac{当前时隙被成功覆盖\text{cluster}数量/\text{cluster}数 +
当前时隙总覆盖\text{UE}数/总\text{UE}数}{系统所有无人机能耗} + 所有非法动作惩罚项 / \text{UAV} 数量
$$

### 1.29

MMFAE：数据量 原始->编码，降低
无人机和HAP传输速率，

reward 重新设计
$$
\frac{服务\text{UE}数}{最大负载} - \frac{能耗}{单时隙最大能耗} + 惩罚项
$$

### 一个比较严重的问题

为什么测试时各个无人机接收到的动作都是一样的？？？
目前感觉UAV的动作输入同质化比较严重

- 将img-class的observation卷积后的维度下降后，目前智能体动作决策同质化现象有改善
- 还有一个是引入了 two_learner 之后，应该也有点效果

## 对奖励设计的一些思考

### reward_total

1. 当智能体产生的非法动作过多时（大于等于一半），得是负的
2. 总能耗不能在奖励中减去，这种操作会使得智能体无法区分是非法动作还是能耗带来奖励的降低。总能耗得放在分母上，总能耗高则奖励变低。
3. 考虑地面UE的总体覆盖程度：
    1. 至当前时隙所有UE覆盖率
    2. 当前时隙被覆盖UE数
    3. 当前时隙被覆盖成功UE cluster数
4. 公平指数

#### 一些思路

1. 极端处理：当智能体有产生非法动作的时候，总奖励就是负的，去掉那些正向的增益

### reward

从无人机个体出发来设计本地的奖励

1. 产生非法动作时，奖励就是负的，就是惩罚项
2. 去掉（或弱化）全局因素的影响
    - 公平指数
    - 当前时隙全局所有UE的覆盖率
3. 智能体更多的只考虑自己（自私的智能体）
    - 服务的用户数
    - 能耗
    - 动作合法性

## 画图一点优化

各个UE的位置用浮点数

## To Do:

- env_2 环境融合
- 补充对训练过程中loss_td的画图功能
- 将 mixing network 代替为 VDN 或 IQL，测试一下结果

## 测试 ID 记录：

- 1043692: 16000000 env_slot, 50 * 50, 6 uav, episode_limit=500, 无人机、cluster 中心初始位置固定
- 3119392：测试vdn，学习率为 0.000001 (1e-5) **效果还可以，有3个UE cluster被覆盖了**
- 2610236：vdn, two_learner

### 失败

- 3964262: 8000000 env_slot, 50\*50, 6 uav, episode_limit=200, 无人机、cluster 中心初始位置固定, transferrable agent,
  全局+本地 reward更新, 两经验缓冲区采样 *失败*
- 1687640: 16005000 env_slot, 50\*50, 6 uav, episode_limit=200, 无人机、cluster 中心初始位置固定, transferrable agent(
  超参数网络+卷积), 全局 reward更新, 两经验缓冲区采样 *失败*
- 2036996：其他配置同1687640，学习率为 0.000000001 (1e-8) *失败*
- 2105262：其他配置同1687640，学习率为 0.0000001 (1e-6) *失败*
- 2379703：降低超参数网络参数大小，学习率为 0.000001 (1e-5) *失败*
- 3548741：vdn、lr=1e-6、TD(0) *失败*

### 测试中

- ustc5 25228：vdn、lr=1e-5、TD(0.4)
- ust5 27161：同 2610236，$\lambda$衰减率为 10000000

设置：$\lambda=5000000$, two_learner, img_conv_dim: 4

### 测试中

- 525955：无人机最高飞行高度4，最大覆盖半径5，UE cluster半径8，cluster覆盖成功阈值0.5，$\lambda$衰减率为 5000000
- ustc5 16009：UE cluster UE 上限更改为25 本地更新
- ustc5 20490：TD(0)

#   M A R L - U A V - n e t w o r k s  
 