### Flask imports
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, Response
)
from flask import Flask
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
from flaskr.auth import login_required
from flaskr.db import get_db

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
            current_video_path = os.path.join(os.path.join(os.getcwd(),'video_files/'),filename)
            file.save(current_video_path)
            print(current_video_path)
            # load_frames_to_buffer(CONFIG["SECONDS_TO_BUFFER"])
            # if current_num_of_frames_buffered > 0:
            return Response(load_frames_to_buffer(CONFIG["SECONDS_TO_BUFFER"]), mimetype= 'multipart/x-mixed-replace; boundary=frame')

    
    return render_template("blog/index.html")


def load_frames_to_buffer(seconds):
    total_frames_to_load = seconds*CONFIG["FRAMERATE"]
    global current_num_of_frames_buffered


    print(f"path: {current_video_path}")
    if current_video_path:
        print("Inside")
        # cap = cv2.VideoCapture(current_video_path)
        cap = cv2.VideoCapture("video_files/video3.mp4")
        print(f"cap:{cap}")
        while cap.isOpened() and current_num_of_frames_buffered < total_frames_to_load:
            ret, frame = cap.read()
            if not ret:
                break
            # frame_buffer.append(buffer)
            ret, buffer = cv2.imencode('.jpeg',frame)
            current_num_of_frames_buffered += 1
            frame = buffer.tobytes()


            yield (b'--frame\r\n' b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n')    
