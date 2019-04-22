import os
from flask import flash, request, url_for, Blueprint, render_template, redirect, Response
from werkzeug.utils import secure_filename
from werkzeug.urls import url_parse
from .forms import LoginForm
from flask_login import current_user, login_user
from flask_login import logout_user
from .models import User
from flask_login import login_required
from .WebApp.Face_Item.video import gen as generate


from .WebApp.Face_Item.camera import VideoCamera as FaceItemCamera
from flask_sapient.WebApp.Posture.camera import VideoCamera as PostureCamera
from flask_sapient.WebApp.Face_Item.video import gen as face_item_video_reading
# from flask_sapient.WebApp.Posture.video import gen as posture_video_reading


main = Blueprint('main', __name__)

UPLOAD_FOLDER = '/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/static/video'
ALLOWED_EXTENSIONS = set(['mp4'])

# main.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# That is the first page of the web
@main.route('/')
@main.route('/index')
@login_required
def index():
    # comments = Comment.query.all()
    return render_template('homepage/index.html')

# login page
@main.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None:
            flash('Invalid username')
            return redirect(url_for('main.login'))
        if not user.check_password(form.password.data):
            flash('Invalid password')
            return redirect(url_for('main.login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('main.index')
        return redirect(next_page)
    return render_template('login.html', form=form)


@main.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main.login'))


@main.route('/imagewebcam')
@login_required
def imagewebcam():
    # comments = Comment.query.all()
    return render_template('image_webcam.html')


@main.route('/videoupload', methods=['GET', 'POST'])
@login_required
def videoupload():
    if request.method == 'POST':
        # check if the post request has the file part
        if request.form['button'] == 'upload':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename('input.mp4')
                file.save(os.path.join(UPLOAD_FOLDER, filename))
        elif request.form['button'] == 'face_item':
            face_item_video_reading()
            return redirect(request.url)
        # elif request.form['button'] == 'posture':
        #     posture_video_reading()
        #     return redirect(request.url)
        elif request.form['button'] == 'analyse':
            text = open('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/video_analyse.txt', 'r+')
            content = text.read().split('\n')
            text.close()
            return render_template('video_upload.html', text=content, video_source=url_for('static', filename='video/output.mp4'))
    return render_template('video_upload.html')


# area for livefeed recognition
@main.route('/livefeed')
@login_required
def livefeed():
    # comments = Comment.query.all()
    return render_template('live_feed.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@main.route('/livefeed_extendent_faceItem_Page')
@login_required
def livefeed_extendent_faceItem_Page():
    # comments = Comment.query.all()
    return render_template('live_feed_extendent/face_item.html')


@main.route('/live_feed_extendent_faceItem_Recogniser')
@login_required
def live_feed_extendent_faceItem_Recogniser():
    return Response(gen(FaceItemCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@main.route('/livefeed_extendent_posture_Page')
@login_required
def livefeed_extendent_posture_Page():
    # comments = Comment.query.all()
    return render_template('live_feed_extendent/posture.html')


@main.route('/live_feed_extendent_posture_Recogniser')
@login_required
def live_feed_extendent_posture_Recogniser():
    return Response(gen(PostureCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@main.route('/livefeed_extendent_analyse')
@login_required
def livefeed_extendent_analyse():
    text = open('/Users/shunshao/Desktop/OpenCV_Web/flask_sapient/WebApp/live_analyse.txt', 'r+')
    content = text.read().split('\n')
    text.close()
    return render_template('live_feed_extendent/analyse.html', text=content)

