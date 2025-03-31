### Flask imports
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, Response, current_app, jsonify
)
from flask import Flask
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
import base64
from flaskr.auth import login_required
from flaskr.db import get_db
import numpy as np
import json
import math


### General libraries
import cv2
import os

ALLOWED_EXTENSIONS = {'mp4','avi'}


CONFIG = {
    "SECONDS_TO_BUFFER" : 30,
    "SECONDS_TO_LOAD_ONE_STEP": 5,
    "FRAMERATE": 30

}


frame_buffer = []
current_frame_buffer_index = 0

current_num_of_frames_buffered = 0
current_video_path = None


bp = Blueprint('blog', __name__)

app = Flask(__name__)



@bp.route('/')
def index():

    return render_template('blog/index.html')

def allowed_file(filename):
    print(filename)
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/video_feed", methods=['GET','POST'])
def video_feed():
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            global current_video_path
            filepath = os.path.join('video_files/',filename)
            
            temp = os.path.join("flaskr/static/",filepath)
            current_app.config["CONTROLLER"]._loadVideo(temp)
            current_video_path = os.path.join(os.getcwd(),temp)
            file.save(current_video_path)
            print(current_video_path)
            fps = get_fps_opencv(filepath)

            numFrames = current_app.config["CONTROLLER"].getNumFrames()

            # load_frames_to_buffer(CONFIG["SECONDS_TO_BUFFER"])
            # if current_num_of_frames_buffered > 0:
            # return Response(load_frames_to_buffer(CONFIG["SECONDS_TO_BUFFER"]), mimetype= 'multipart/x-mixed-replace; boundary=frame')
            return render_template("blog/annotator.html",video_filepath = filepath, framerate=fps, numFramesVideo = numFrames)
    
    return render_template("blog/index.html")

def get_fps_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # Retrieve FPS
    cap.release()
    return fps


def convert_coordinates(x, y, src_res, dst_res, src_padding=(0, 0, 0, 0), dst_padding=(0, 0, 0, 0)):
    """
    Converts the coordinates (x, y) from one resolution to another, accounting for padding.
    
    :param x: X-coordinate in the source resolution
    :param y: Y-coordinate in the source resolution
    :param src_res: Tuple (width, height) of the source resolution
    :param dst_res: Tuple (width, height) of the destination resolution
    :param src_padding: Tuple (left, top, right, bottom) padding in the source resolution
    :param dst_padding: Tuple (left, top, right, bottom) padding in the destination resolution
    :return: Transformed (x, y) coordinates in the destination resolution
    """
    # Extract padding values
    src_left, src_top, src_right, src_bottom = src_padding
    dst_left, dst_top, dst_right, dst_bottom = dst_padding
    
    # Compute effective source and destination dimensions (excluding padding)
    effective_src_width = src_res[0] - (src_left + src_right)
    effective_src_height = src_res[1] - (src_top + src_bottom)
    
    effective_dst_width = dst_res[0] - (dst_left + dst_right)
    effective_dst_height = dst_res[1] - (dst_top + dst_bottom)
    
    # Normalize coordinates relative to the effective source dimensions
    norm_x = (x - src_left) / effective_src_width
    norm_y = (y - src_top) / effective_src_height
    
    # Convert to destination coordinates
    new_x = norm_x * effective_dst_width + dst_left
    new_y = norm_y * effective_dst_height + dst_top
    
    return new_x, new_y

# Create a function to scale a list of points from one resolution to another taking into account padding
def scale_points_list(data, input_res, output_res, padding=(0, 0, 0, 0)):
    scaled_data = [convert_coordinates(x, y, input_res,output_res,padding) for x,y in data]
    return scaled_data

def scale_points(data, input_res, output_res, padding=(0, 0, 0, 0)):
    """
    Scale points from one resolution to another considering padding.

    :param data: Dictionary with keys mapping to lists of (x, y) points
    :param input_res: Tuple (width, height) of input resolution
    :param output_res: Tuple (width, height) of output resolution
    :param padding: Tuple (left, top, right, bottom) specifying padding in output resolution
    :return: Dictionary with scaled points
    """
    input_w, input_h = input_res
    output_w, output_h = output_res
    pad_left, pad_top, pad_right, pad_bottom = padding

    # Compute effective drawing area in output resolution
    effective_w = output_w - pad_left - pad_right
    effective_h = output_h - pad_top - pad_bottom

    scale_x = effective_w / input_w
    scale_y = effective_h / input_h

    scaled_data = {}
    # print(data.items())
    for key, points in data.items():
        print(points)
        print(points[0])
        print(f"key: {key}")
        scaled_data[key] = []
        for mask in points:
            scaled_data[key].append([(pad_left + x * scale_x, pad_top + y * scale_y) for x, y in mask])

    return scaled_data


@bp.route("/apply_sam", methods=["GET","POST"])
def apply_sam():
    # print(f"datos del request son: {request}")
    if request.method == "POST":
        print("SAM applying")
        valores = request.form

        actual_frame = valores["current_frame"]
        verticalPadding = int(valores["vPad"])
        puntos = json.loads(valores["puntos"])
        print(valores)

        srcRes = (int(valores["crW"]),int(valores["crH"]))
        dstRes = (int(valores["vrW"]),int(valores["vrH"]))

        for frame,frame_points in enumerate(puntos):

            if frame == int(actual_frame):
                # print(f"{actual_frame} : {frame} : {frame_points}" )
                for point in frame_points:
                    # print(f"puntos: {len(point)} - puntos: {point} - {actual_frame}")
                    if point: 
                        # vertical_point = math.floor((point[1]-int(verticalPadding)) * (1+scale))
                        # horizontal_point = math.floor(point[0] * (1+scale))
                        new_point = convert_coordinates(point[0],point[1],srcRes,dstRes,(0,0,0,0))
                        print(f"new point : {new_point}")
                        current_app.config["CONTROLLER"]._addPoint(new_point, 0, str(int(actual_frame)))

        # model_points = current_app.config["CONTROLLER"].getPoints()
        # print(f"image: {model_points}")
        # print(f"image: {current_app.config['CONTROLLER']}")
        # print(f"image: {current_app.config['CONTROLLER'].getModel().currentPoints}")
        samPred = current_app.config["CONTROLLER"].applySAM(frame=int(actual_frame))

        # current_app.config["CONTROLLER"].propagate()
        listOfPolygons = []
        for mask in samPred:
            # print(f"\nmask: {mask.getMask()}")
            listOfPolygons.append(mask.getMask())
        # print(listOfPolygons)
        # current_app.config["CONTROLLER"].propagate()  
        return jsonify(listOfPolygons)

@bp.route("/apply_cutie", methods=["GET","POST"])
def apply_cutie():
    if request.method == "POST":
        print("Cutie applying")
        valores = request.form

        dstRes = (int(valores["crW"]),int(valores["crH"]))
        srcRes = (int(valores["vrW"]),int(valores["vrH"]))

        current_app.config["CONTROLLER"].propagate()
        all_current_masks = current_app.config["CONTROLLER"].getAllMasks()

        print(f"masks structure: {all_current_masks}")
        print(f"masks typing: {type(all_current_masks)}")


        dictMasks = {}

        for key, masks in all_current_masks.items():
            for mask in masks:
                if (key) in dictMasks:
                    dictMasks[key].append(mask.getMask())
                else:
                    dictMasks[key] = [mask.getMask()]

        dictMasks = scale_points(dictMasks, srcRes, dstRes, (0,0,0,0))
        return jsonify(dictMasks)


@bp.route("/clear_masks", methods=["GET","POST"])
def clear_masks():
    if request.method == "POST":
        current_app.config["CONTROLLER"].clearAll()

        temp = {'1':1}
        return jsonify(temp)

@bp.route("/save_ann", methods=["GET","POST"])
def save_ann():
    if request.method == "POST":
        current_app.config["CONTROLLER"].save_ann()

        temp = {'1':1}
        return jsonify(temp)

@bp.route("/add_points_to_mask", methods=["GET","POST"])
def add_points_to_mask():
    if request.method == "POST":
        valores = request.form
        srcRes = (int(valores["crW"]),int(valores["crH"]))
        dstRes = (int(valores["vrW"]),int(valores["vrH"]))

        print(valores)
        maskToAdd = json.loads(valores["maskToAdd"])
        print(type(maskToAdd))
        print(maskToAdd)
        current_app.config["CONTROLLER"].addCorrectionToMask(int(valores["maskIndex"]),scale_points_list(maskToAdd, srcRes, dstRes, (0,0,0,0)))
        temp = {'1':1}
        return jsonify(temp)

@bp.route("/get_masks", methods=["GET","POST"])
def get_masks():
    if request.method == "POST":
        all_current_masks = current_app.config["CONTROLLER"].getAllMasks()
        print(f"masks structure: {all_current_masks}")
        print(f"masks typing: {type(all_current_masks)}")
        valores = request.form
        dstRes = (int(valores["crW"]),int(valores["crH"]))
        srcRes = (int(valores["vrW"]),int(valores["vrH"]))
        dictMasks = {}

        for key, masks in all_current_masks.items():
            for mask in masks:
                if (key+1) in dictMasks:
                    dictMasks[key+1].append(mask.getMask())
                else:
                    dictMasks[key+1] = [mask.getMask()]

        dictMasks = scale_points(dictMasks, srcRes, dstRes, (0,0,0,0))
        return jsonify(dictMasks)
    
@bp.route("/get_video_frames", methods=["GET","POST"])
def get_video_frames():
    if request.method == "POST":
        frames = current_app.config["CONTROLLER"].getFrames()
        print(f"frames: {frames}")
        return jsonify(frames)
    
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

@bp.route('/get_frames', methods=['GET', 'POST'])
def get_frames():
    if request.method == 'POST':
        frames = []
        all_frames = current_app.config["CONTROLLER"].getFrames() 

        for frame in all_frames:
            frame_base64 = frame_to_base64(frame)
            frames.append(frame_base64)

        return jsonify(frames=frames)