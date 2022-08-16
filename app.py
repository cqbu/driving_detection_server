from flask import Flask, request, send_from_directory, jsonify, Response
from task import Task, TaskList
from concurrent.futures import ThreadPoolExecutor
import os
import json
import cv2
import sys
sys.path.append('./yolov5')
sys.path.append('./CLRNet')
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from CLRNet.infer import build_model,get_sample,show_lanes
from CLRNet.clrnet.utils.config import Config
import numpy as np
import torch 
#load model
device=''
device = select_device(device)
executor = ThreadPoolExecutor(8)
app = Flask(__name__)
tl = TaskList()
submitdata={
    "namesofvideo":[], 
    "weightsofyolov5":[],
    "weightsofCLR":[]
}
backbones_configs={
    "Resnet18":"./CLRNet/configs/clrnet/clr_resnet18_culane.py",
    "m3s":"./CLRNet/configs/clrnet/clr_mobilenetv3s_culane.py",
    "m3l":"./CLRNet/configs/clrnet/clr_mobilenetv3l_culane.py"
}
namesofvideo =[];

@app.route('/index')
def index():
    return 'Flask Index'

# 上传视频
@app.route('/upload/video', methods=['POST'])
def upload_video():
    file = request.files.get('file')
    file.save(os.path.join('./videos', file.filename))
    video = cv2.VideoCapture(os.path.join('./videos',file.filename))
    ret,frame = video.read()
    cv2.imwrite(os.path.join("./videos",file.filename.split('.')[0]+".jpg"),frame)
    video.release()
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

# 查询已存在信息并返回
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

def run_task(task: Task):
    try:
        task.set_status('running')
        config = task.get_info()
        yolov5_path = os.path.join('./weights/yolov5',config['yolov5_model_name'])
        try:
            yolo_model = DetectMultiBackend(yolov5_path, device=device, dnn=False, data="./yolov5/data/bdd100k.yaml")
        except Exception:
            raise Exception(f'无法载入yolov5模型：{yolov5_path}')
        video_path = os.path.join('./videos',config['video_name'])
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception(f'无法打开视频: {video_path}')
        fps = video.get(cv2.CAP_PROP_FPS)
        video_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output = cv2.VideoWriter(os.path.join('output',config['name']+'_' + config['video_name']), cv2.VideoWriter_fourcc(*'XVID'),
                                fps, video_size)
        num_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        yolo_model.model.float()
        clr_config = cfg = Config.fromfile(backbones_configs[config["clrnet_backbone"]])
        print ("buidingclr_model")
        clr_path = os.path.join("./weights/CLRNet",config['clrnet_model_name'])
        try:
            clr_model,clr_cfg=build_model(backbones_configs[config["clrnet_backbone"]], clr_path)
        except Exception:
            raise Exception(f'无法载入CLRNet模型：{clr_path}')
        print("builded clrmodel")
        cnt_yolo = 0
        cnt_clr = 0 
        while video.isOpened():
            ret  , frame =  video.read()
            if not ret:
                break
            if cnt_yolo % config['yolov5_period'] == 0:
                pred,stride,pt=yolosingelimage(frame,yolo_model)
            if cnt_clr % config['clrnet_period'] == 0:
                lanes = clr_inference_single_image(frame,clr_model,clr_config)
            frame = addboxes(pred,frame,yolo_model)
            sample = get_sample(frame,clr_cfg)
            frame = show_lanes(sample,lanes)
            output.write(frame)
            cnt_yolo = cnt_yolo + 1
            cnt_clr = cnt_clr +1
            task.set_progress(cnt_yolo / num_frame)
            
            # for i in range(config['yolov5_period']-1):
            #     ret  , frame =  video.read()
            #     if not ret:
            #         break
            #     frame = addboxes(pred,frame,model)
            #     output.write(frame)
        video.release()
        output.release()
        task.set_status('done')
    except Exception as e:
        task.set_status('fail')
        task.set_err_msg(str(e))
        print(task.get_info())
    
def clr_inference_single_image(img, model, cfg):
    sample = get_sample(img, cfg)
    with torch.no_grad():
        output = model(sample)
        output = model.module.heads.get_lanes(output)
    lanes = [lane.to_array(cfg) for lane in output[0]]
    return lanes
def yolosingelimage(_img , model):
    img = _img.copy()
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size([640,640], s=stride)  # check image size
    model.eval()
    assert isinstance(model, DetectMultiBackend)
    img0,x,(dw,dh)= letterbox(img,imgsz,stride=stride,auto=pt)
    img0 = img0.transpose((2, 0, 1))[::-1]
    img0 = np.ascontiguousarray(img0)
    img0 = torch.from_numpy(img0).to(model.device)
    img0 = img0.float()
    img0 /= 255
    img0 = img0.unsqueeze(0)
    pred=model(img0,augment=False, visualize=False)
    print('type(pred): ', type(pred))
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45,
    classes=None, agnostic=False, max_det=1000)
    return pred,stride,pt
def addboxes(pred,img,model):
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size([640,640], s=stride) 
    img0,x,(dw,dh)= letterbox(img,imgsz,stride=stride,auto=pt)
    img0 = img0.transpose((2, 0, 1))[::-1]
    img0 = np.ascontiguousarray(img0)
    img0 = torch.from_numpy(img0).to(model.device)
    img0 = img0.float()
    img0 /= 255
    img0 = img0.unsqueeze(0)
    for i,det0 in enumerate(pred):
        det = det0.clone()
        img_=img.copy()
        annotator = Annotator(img_, line_width=3, example=str(names))
        if len(det):
            det[:,:4] = scale_coords(img0.shape[2:],det[:,:4],img_.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label =  f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
        img_=annotator.result()
    return img_
    
# 提交任务配置
@app.route('/submit', methods=['POST'])
def submit_task():
    config = request.get_data()
    config = json.loads(config)
    tl.add(config)
    print(config)
    executor.submit(run_task, tl[-1])
    # run_task(tl[-1])
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
            chunk = f.read(20 * 1024 * 1024)
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