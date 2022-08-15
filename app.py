from flask import Flask, request, send_from_directory, jsonify, Response
from task import Task, TaskList
from concurrent.futures import ThreadPoolExecutor
import os
import json
import cv2

executor = ThreadPoolExecutor(8)
app = Flask(__name__)
tl = TaskList()
submitdata={
    "namesofvideo":[],
    "weightsofyolov5":[],
    "weightsofCLR":[]
};
namesofvideo =[];

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

# 查询已存在图片信息并返回
@app.route('/videolist', methods=['GET'])
def get_videolist():
    datanames=os.listdir("./videos")
    datanames.sort()
    submitdata={
    "namesofvideo":[],
    "weightsofyolov5":[],
    "weightsofCLR":[]
};
    for dataname in datanames:
        if os.path.splitext(dataname)[1] == '.jpg':
            submitdata['namesofvideo'].append(dataname.split(".")[0]+".mp4")
    datanames=os.listdir("./weights/yolov5")
    datanames.sort()
    for dataname in datanames:
            submitdata['weightsofyolov5'].append(dataname)
    datanames=os.listdir("./weights/CLRNet")
    datanames.sort()
    for dataname in datanames:
            submitdata['weightsofCLR'].append(dataname)
    print(submitdata)
    return jsonify(submitdata)

#显示图片
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
    info = tl[task_id].get_info()
    return jsonify(info)

# 查询所有任务配置信息
@app.route('/tasklist', methods=['GET'])
def get_all_task_configs():
    infos = [task.get_info() for task in tl]
    return jsonify(infos)

# 删除任务
@app.route('/taskdelete/<int:task_id>')
def delete_task(task_id):
    tl.remove(task_id)
    return 'Delete successfully'

@app.route('/taskrun/<int:task_id>')
def task_run(task_id):
    task = tl[task_id]
    executor.submit(task.run)
    return 'done'

def generate_frames(video_path):
    video = cv2.VideoCapture(video_path)
    while True:
        success, frame = video.read()
        if success:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        else:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 在线视频流
@app.route('/onlinevideo/<string:video_name>')
def online_video(video_name):
    video_path = os.path.join('output', video_name)
    return Response(generate_frames(video_path=video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_download_stream(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(3 * 1024 * 1024)
            if not chunk:
                break
            yield chunk

# 下载输出文件
@app.route('/downloadoutput/<string:video_name>')
def download_output(video_name):
    video_path = os.path.join('output', video_name)
    response = Response(generate_download_stream(video_path), content_type="application/octet-stream")
    response.headers['content-length'] = os.stat(str(video_path)).st_size
    return response

if __name__ == '__main__':
    if not os.path.exists('./videos'):
        os.makedirs('./videos')
    if not os.path.exists('./output'):
        os.makedirs('./output')
    if not os.path.exists('./weights/CLRNet'):
        os.makedirs('./weights/CLRNet')
    if not os.path.exists('./weights/yolov5'):
        os.makedirs('./weights/yolov5')
    
    app.run(host="0.0.0.0", port=5000, debug=True)