from flask import Flask, request
import os

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

if __name__ == '__main__':
    if not os.path.exists('./videos'):
        os.makedirs('./videos')
    if not os.path.exists('./weights/CLRNet'):
        os.makedirs('./videos')
    if not os.path.exists('./weights/yolov5'):
        os.makedirs('./videos')
    
    app.run(host="0.0.0.0", port=5000, debug=True)