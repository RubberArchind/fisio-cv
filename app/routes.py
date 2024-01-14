import base64
import cv2 as cv
import numpy as np
from flask import render_template, Response, jsonify, request
from PIL import Image
from controllers.craniovertebra_angle import CraniovertebraAngle
from controllers.forward_shoulder_angle import ForwardShoulderAngle
from controllers.carrying_angle import CarryingAngle
from controllers.q_angle import QAngle
from controllers.camera import Record
from io import BytesIO
import time

class Routes:
    def __init__(self, app):
        self.app = app
        self.cv = CraniovertebraAngle()
        self.fsa = ForwardShoulderAngle()
        self.carry = CarryingAngle()
        self.q = QAngle()

    def setup(self):
        self.index()
        self.craniovertebra()
        self.forward_shoulder()
        self.carrying()
        self.q_angle()

    # Home
    def index(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

    # Craniovertebra Angle
    def craniovertebra(self):
        @self.app.route('/craniovertebra')
        def craniovertebra():
            return render_template('craniovertebra.html')

        @self.app.route('/craniovertebra_upload', methods=["POST"])
        def craniovertebra_upload():
            file = Image.open(BytesIO(base64.b64decode(request.form['image'])))
            filename = "/tmp/{}".format(time.time())
            print(filename)
            file.save(filename, 'PNG')
            return Response(self.cv.run(filename), mimetype='application/json')

        @self.app.route('/craniovertebra_vid')
        def craniovertebra_vid():
            return Response(self.cv.run(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/record_cv')
        def record_cv():
            res = self.cv.results
            if res:
                Record.save_result('craniovertebra', res)
                return jsonify("success")
            else:
                return jsonify("failed")

    # Forward Shoulder Angle
    def forward_shoulder(self):
        @self.app.route('/forward_shoulder')
        def forward_shoulder():
            return render_template('forward_shoulder.html')

        @self.app.route('/forward_shoulder_upload', methods=["POST"])
        def forward_shoulder_upload():
            file = Image.open(BytesIO(base64.b64decode(request.form['image'])))
            filename = "/tmp/{}".format(time.time())
            print(filename)
            file.save(filename, 'PNG')
            return Response(self.fsa.run(filename), mimetype='application/json')

        @self.app.route('/forward_shoulder_vid')
        def forward_shoulder_vid():
            return Response(self.fsa.run(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/record_fsa')
        def record_fsa():
            res = self.fsa.results
            if res:
                Record.save_result('forward_shoulder', res)
                return jsonify("success")
            else:
                return jsonify("failed")

    def carrying(self):
        @self.app.route('/carrying')
        def carrying():
            return render_template('carrying.html')

        @self.app.route('/carrying_upload', methods=["POST"])
        def carrying_upload():
            file = Image.open(BytesIO(base64.b64decode(request.form['image'])))
            filename = "/tmp/{}".format(time.time())
            print(filename)
            file.save(filename, 'PNG')
            return Response(self.carry.run(filename), mimetype='application/json')

        @self.app.route('/carrying_vid')
        def carrying_vid():
            return Response(self.carry.run(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/record_carry')
        def record_carry():
            res = self.carry.results
            if res:
                Record.save_result('carrying', res)
                return jsonify("success")
            else:
                return jsonify("failed")

    def q_angle(self):
        @self.app.route('/q_angle')
        def q_angle():
            return render_template('q_angle.html')

        @self.app.route('/q_angle_upload', methods=["POST"])
        def q_angle_upload():
            file = Image.open(request.files['image'].stream)
            print(file)
            rbga_file = file.convert('RGB')
            print(rbga_file)
            rbga_file.save("tes.jpg")
            return Response(self.q.run('tes.jpg'), mimetype='application/json')

        @self.app.route('/q_angle_vid')
        def q_angle_vid():
            return Response(self.q.run(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/record_q')
        def record_q():
            res = self.q.results
            if res:
                Record.save_result('q_angle', res)
                return jsonify("success")
            else:
                return jsonify("failed")
