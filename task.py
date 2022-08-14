"""
config是一个dict，包含以下信息：
    - name: str，任务名称
    - video_name: str，不含路径的视频文件名
    - yolov5_model_name：str，不含路径的yolov5模型文件名
    - clrnet_model_name：str，不含路径的clrnet模型文件名
    - clrnet_backbone：str，backbone名称，包括'r18'，'r34'，'m3s'，'m3l'等
    - yolov5_period：int，yolov5处理周期，即每period帧处理一次，默认值为1
    - clrnet_period：int，clrnet处理周期，即每period帧处理一次，默认值为1
"""

class Task:
    def __init__(self, config: dict):
        self.__config = config
        self.__status = 'waiting' # waiting, running, done
    
    def run(self):
        self.__status = 'running'
        
        self.__status = 'done'

    def get_config(self):
        return self.__config
    
    def get_status(self):
        return self.__status
        

class TaskList:
    def __init__(self):
        self.__list = []
        self.__name_dict = {}
    
    # 添加任务
    def add(self, config: dict):
        task = Task(config)
        self.__name_dict[config['name']] = task
        self.__list.append(task)
    
    # 删除任务，key可以为任务名称(str)或下标(int)
    def remove(self, key):
        if isinstance(key, int):
            task = self.__list.pop(key)
            name = task.get_config()['name']
            self.__name_dict.pop(name)
        elif isinstance(key, str):
            task = self.__name_dict.pop(key)
            self.__list.remove(task)
        else:
            raise TypeError
            
    # 从TaskList中取出任务，key可以为任务名称(str)或下标(int)
    def __getitem__(self, key) -> Task:
        if isinstance(key, int):
            return self.__list[key]
        elif isinstance(key, str):
            return self.__name_dict[key]
        else:
            raise TypeError

    # 获取TaskList长度
    def __len__(self):
        return len(self.__list)
        