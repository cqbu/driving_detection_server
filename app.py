from flask import Flask, request, send_from_directory, jsonify
from task import Task, TaskList
from concurrent.futures import ThreadPoolExecutor
import os
import json

executor = ThreadPoolExecutor(8)
app = Flask(__name__)
tl = TaskList()

@app.route('/index')
def index():
    return 'Flask Index'

# 上传视频
@app.route('/upload/video', methods=['POST'])
def upload_video():
    file = request.files.get('file')
    file.save(os.path.join('./videos', file.filename))
    return 'Upload successfully'

# 上传物体检测模型
@app.route('/upload/od_model', methods=['POST'])
def upload_od_model():
    file = request.files.get('file')
    file.save(os.path.join('./weights/yolov5', file.filename))
    return 'Upload successfully'

# 上传车道检测模型
@app.route('/upload/ld_model', methods=['POST'])
def upload_ld_model():
    file = request.files.get('file')
    file.save(os.path.join('./weights/CLRNet', file.filename))
    return 'Upload successfully'

@app.route('/download/<string:image_name>', methods=['GET'])
def download(image_name):
    image_name = os.path.basename(image_name)
    image_name=image_name.split(".")[0]
    image_name=image_name+".jpg"
    if os.path.isfile(os.path.join("videos", image_name)):
        return send_from_directory("videos", image_name)
    pass

# 提交任务配置
@app.route('/submit', methods=['POST'])
def submit_task():
    config = request.get_data()
    config = json.loads(config)
    tl.add(config)
    return 'submitted'

# 查询某个任务配置信息
@app.route('/tasklist/<int:task_id>', methods=['GET'])
def get_task_config(task_id):
    config = tl[task_id].get_info()
    return jsonify(config)

# 查询所有任务配置信息
@app.route('/tasklist', methods=['GET'])
def get_all_task_configs():
    configs = [task.get_info() for task in tl]
    return jsonify(configs)

@app.route('/taskrun/<int:task_id>')
def task_run(task_id):
    task = tl[task_id]
    executor.submit(task.run)
    return 'done'

if __name__ == '__main__':
    if not os.path.exists('./videos'):
        os.makedirs('./videos')
    if not os.path.exists('./weights/CLRNet'):
        os.makedirs('./weights/CLRNet')
    if not os.path.exists('./weights/yolov5'):
        os.makedirs('./weights/yolov5')
    
    app.run(host="0.0.0.0", port=5001, debug=True)