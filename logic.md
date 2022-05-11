# utils.py
- make **`dataset`**, define **`optimizer`** for training
## create_buffers
- generate three buffers for the three positions
- distinguish between `gpu & cpu`
    ```python
    buffers[device][position]=_buffers
    ```
- **TO**: `get_batch`
## act
- run **forever**
- get data from `environment`
  ```python
  env = create_env(flags)
  env = Environment(env, device)
  position, obs, env_output = env.initial()
  position, obs, env_output = env.step(action)
  ```
- **`buffer`**: store data
- **`full_queue`** **`free_queue`**: sync-up with main process
# dmc.py
# env_utils.py

### create_buffers
**input**: 
- length of time dimension
- different device

为每个设备的每个位置创建一个**空的**容器，存放state, information等信息

**return**:
- buffer 容器

**`share memory`**