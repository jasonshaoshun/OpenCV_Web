import os
from flask import flash, request, url_for, Blueprint, render_template, redirect, Response
from werkzeug.utils import secure_filename
from .forms import LoginForm
from flask_login import current_user, login_user
from .models import User

from .WebApp.camera import VideoCamera

main = Blueprint('main', __name__)

UPLOAD_FOLDER = '/Users/shunshao/Desktop/team/flask_web'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'])

# main.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main.route('/')
def index():
    # comments = Comment.query.all()
    return render_template('homepage/index.html')


@main.route('/imageupload', methods=['GET', 'POST'])
def imageupload():
    if request.method == 'POST':
        # check if the post request has the file part
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
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
    return render_template('image_upload.html')


@main.route('/imagewebcam')
def imagewebcam():
    # comments = Comment.query.all()
    return render_template('image_webcam.html')


@main.route('/videoupload')
def videoupload():
    # comments = Comment.query.all()
    return render_template('video_upload.html')


@main.route('/livefeed')
def livefeed():
    # comments = Comment.query.all()
    return render_template('live_feed.html')


# @main.route('/login', methods=['GET', 'POST'])
# def login():
#     form = LoginForm()
#     if form.validate_on_submit():
#         flash('Login requested for user {}, remember_me={}'.format(
#             form.username.data, form.remember_me.data))
#         return redirect(url_for('main.index'))
#     return render_template('login.html', form=form)


@main.route('/login', methods=['GET', 'POST'])
def login():
    # if current_user.is_authenticated:
    #     return redirect(url_for('main.index'))
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
        return redirect(url_for('main.index'))
    return render_template('login.html', form=form)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@main.route('/video_feed')
def live_cam():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# @main.route('/sign')
# def sign():
#     return render_template('sign.html')
#
# @main.route('/sign', methods=['POST'])
# def sign_post():
#     name = request.form.get('name')
#     comment = request.form.get('comment')
#
#     new_comment = Comment(name=name, comment_text=comment)
#     db.session.add(new_comment)
#     db.session.commit()
#
#     return redirect(url_for('main.index'))

