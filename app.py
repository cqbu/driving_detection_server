from flask import Flask, request, send_from_directory
import os
import json

app = Flask(__name__)

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

@app.route('/submit', methods=['POST'])
def submit_task():
    data = request.get_data()
    data = json.loads(data)
    print(data)
    return 'submitted'

if __name__ == '__main__':
    if not os.path.exists('./videos'):
        os.makedirs('./videos')
    if not os.path.exists('./weights/CLRNet'):
        os.makedirs('./weights/CLRNet')
    if not os.path.exists('./weights/yolov5'):
        os.makedirs('./weights/yolov5')
    
    app.run(host="0.0.0.0", port=5001, debug=True)