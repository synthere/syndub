# -*- coding: utf-8 -*-
import sys, os

#pyside6-uic.exe  -o .\mainwin.py .\mainwin.ui
from mainwin import Ui_MainWindow

from PySide6.QtCore import QSize, Qt, QUrl, Signal, QTimer, QThread
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QWidget
from PySide6 import QtWidgets
import config
from typing import Iterator, Union
#pyinstaller dubbing.spec

os.environ['QT_MEDIA_BACKEND'] = 'windows'
import dub
import dubref
import subprocess
from functools import wraps

# save original function
__old_Popen = subprocess.Popen

# create wrapper to be called instead of original one
@wraps(__old_Popen)
def new_Popen(*args, startupinfo=None, **kwargs):
    if startupinfo is None:
        startupinfo = subprocess.STARTUPINFO()

    # way 1, as SO suggests:
    # create window
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    # and hide it immediately
    startupinfo.wShowWindow = subprocess.SW_HIDE

    # way 2
    #startupinfo.dwFlags = subprocess.CREATE_NO_WINDOW
    return __old_Popen(*args, startupinfo=startupinfo, **kwargs)

# monkey-patch/replace Popen
subprocess.Popen = new_Popen

def get_base_path():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return base_path

class TranslatorThread(QThread):

    signal_progress_update = Signal(list)
    signal_timer = Signal(str)

    # mod= 1 role is used as audiofile
    def __init__(self, mod, input_file, source_lang, target_lang, role, output_path, speak_num):
        super().__init__() #TranslatorThread, self
        self.mod = mod
        self.inputfile = input_file
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.role = role
        self.speaker_num = speak_num
        self.outputfilePath = output_path
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(1000)  # 1000ms
        self.dubbing = dub.VideoDubbing()
        self.dubref = dubref.VideoDubRef()
        self.dub_stat = 0
    def run(self):
        #self.dub_stat = 1
        if 0 == self.mod:
            self.dubbing.trans(self.inputfile, self.source_lang, self.target_lang, self.role, self.outputfilePath, self.speaker_num)
            config.logger.info('tts mode dub finish')
        else:
            #self.role = './bak/getric_vall.wav'
            if not os.path.exists(self.role):
                config.logger.info('audio mode no ref audio')
                return
            self.dubref.trans(self.inputfile, self.source_lang, self.target_lang, self.role, self.outputfilePath)
            config.logger.info('audio mode finish')

        self.signal_timer.emit("finish")
        #self.dub_stat = 0
    def update_progress(self):
        if 0 == self.mod:
            percent = self.dubbing.translate_percent
        else:
            percent = self.dubref.translate_percent
        #print("to emiit:", percent)
        self.signal_progress_update.emit([percent, 100])

    def finish(self):
        self.signal_progress_update.emit([100, 100])
        self.timer.stop()

from regwin import Ui_RegisterWin
class RegWindows(QWidget, Ui_RegisterWin):
    reg_closed = Signal(str)
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        #self.show()
        self.sel_lic_file.pressed.connect(self.sel_lic)
        self.activat_but.pressed.connect(self.activate_lic)
        self.lic_file = ''
        self.reg_stat = False
    def sel_lic(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, 'Select audio file', '', 'Licence File(*.bin);;All Files (*)',
                                              options=options)
        self.sel_lic_file = file

    def closeEvent(self, event):
        print("widget is closing")
        self.reg_closed.emit("closed")
    def activate_lic(self):
        from register import LicRegister

        lic_num = self.lic_code_txt.toPlainText()
        if lic_num != "":
            lic = lic_num
            print("lic:", lic)
        elif self.sel_lic_file != "":
            with open(self.sel_lic_file, 'r') as fp:
                lic = fp.read()
                # validate
        else:
            return
        lr = LicRegister()
        reg_file = 'reg.txt'

        res = lr.register(lic, reg_file)
        if res:
            self.reg_stat = True
            QMessageBox.warning(None, "Sucess", "Registered sucessfully!")
        else:
            QMessageBox.warning(None, "Fail", "Register failed!")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.inputfile = ''
        self.outputfilePath = ''
        self.selected_ref_audio = ''
        self.use_tts = True
        self.use_ref_audio = False

        self.startbut.pressed.connect(self.startdub)
        self.select_inputpath.pressed.connect(self.select_input_file)
        self.select_savepath.pressed.connect(self.save_out_file)
        self.select_audio_file_but.pressed.connect(self.select_audio_file)
        self.preview_role.pressed.connect(self.preview_role_voice)

        self.radio_use_tts.toggled.connect(self.set_use_tts_stat)
        self.radio_use_audio.toggled.connect(self.set_use_audio_stat)
        self.radio_use_tts.setChecked(True)
        self.radio_use_audio.setChecked(False)

        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Help")

        reg_action = help_menu.addAction("Register")
        about_action = help_menu.addAction("About")

        reg_action.triggered.connect(self.show_reg_window)
        about_action.triggered.connect(self.show_about_dialog)

        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        #self.translate_process_bar.setFixedWidth(0)
        self.translate_process_bar.hide()
        self.unlimit_use = False # limit file length
        self.check_reg_stat()
        self.reg_win = RegWindows()
        self.reg_win.reg_closed.connect(self.reg_reset)
        self.show()
    def reg_reset(self):
        print("reg windows closed, reg res:", self.reg_win.reg_stat)
        if self.reg_win.reg_stat:
            self.unlimit_use = True
    def check_reg_stat(self):
        from register import LicRegister
        lr = LicRegister()
        reg_file = 'reg.txt'
        auth = lr.checkAuthored(reg_file)
        if auth:
            print("unlimit use")
            self.unlimit_use = True
    def show_reg_window(self):

        self.reg_win.show()
        self.reg_win .setWindowTitle("Register")

    def set_use_tts_stat(self):
        self.use_tts = True
        self.use_ref_audio = False
        #self.radio_use_tts.setChecked(True)
        #self.radio_use_audio.setChecked(False)

    def set_use_audio_stat(self):
        self.use_tts = False
        self.use_ref_audio = True
        #self.radio_use_tts.setChecked(False)
        #self.radio_use_audio.setChecked(True)

    def get_current_role(self):
        return [self.role1.currentText(), self.role2.currentText()]

    def get_current_speaker_num(self):
        sn = self.speaknum_box.currentText()
        return int(sn)

    def get_current_source_lang(self):
        return self.origin_lang.currentText()

    def get_current_target_lang(self):
        return self.target_lang.currentText()

    def get_current_translator(self):
        return self.translator.currentText()
    def preview_role_voice(self):
        # play mp3 file
        cur_role = self.role1.currentText()#self.get_current_role()
        wav_path = "./res/"
        if cur_role == "zh-CN-YunjianNeural":
            wav_path += "magic-yunjian.wav"
        elif cur_role == "zh-CN-YunxiNeural":
            wav_path += "magic-yunxi.wav"
        elif cur_role == "zh-CN-XiaoyiNeural":
            wav_path += "magic-xiaoyi.wav"
        elif cur_role == "zh-CN-XiaoxiaoNeural":
            wav_path += "magic-xiaoxiao.wav"
        wav_path = os.path.join(get_base_path(), wav_path)
        #effect = QSoundEffect()
        print("wav path:", wav_path)

        self.player.setAudioOutput(self.audio)
        self.player.setSource(QUrl.fromLocalFile(wav_path))
        self.player.play()

    def show_about_dialog(self):
        about_text = "<h3>Synthere Dub 1.0</h3>" \
                     "<p style='font-size: 12px;'>Copyright © 2024 <a href='https://www.synthere.com'>Synthere</a></p>"
                     #"<ul style='font-size: 14px;'>" \
                     #"<li>More info <a href='https://www.example.com'>Synthere</a></li>" \
                     #"</ul>"
        if self.unlimit_use:
            about_text +="<p style='font-size: 12px;'>Registerd Version. </p>"
        else:
            about_text += "<p style='font-size: 12px;'>Preview Version. Support file less than 30s.</p>"

        about_box = QMessageBox()
        about_box.setWindowTitle("About")
        about_box.setTextFormat(Qt.AutoText)
        about_box.setText(about_text)

        about_box.exec()
    def save_out_file(self):
        options = QFileDialog.Options()
        #file, _ = QFileDialog.getSaveFileName(self, 'Save file', '', 'Video File(*.mp4 *.avi *.mov *.mpg *.mkv);;All Files (*)', options=options)
        file = QFileDialog.getExistingDirectory(self, "Select path")
        print("set output file:", file)
        if "" == file.strip():
            return
        self.outputfilePath = file
        self.savepath.setText(': ' + file) #os.path.basename(file)
    def select_input_file(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, 'Select audio file', '', 'Video File(*.mp4);;All Files (*)',
                                              options=options)
        self.inputfile = file
        self.inputpath_2.setText(': ' + file)#os.path.basename(file)

    def select_audio_file(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, 'Select audio file', '', 'Audio File(*.wav *.mp3);;All Files (*)',
                                              options=options)
        self.selected_ref_audio = file
        self.ref_audio_file_lab.setText(': ' + file)

    def get_file_duration(self, video_file):
        from moviepy.editor import VideoFileClip
        vc = VideoFileClip(video_file)
        dur = vc.duration
        print("file dur:", dur)
        vc.close()
        return dur
    def startdub(self):
        if '' == self.outputfilePath or '' == self.inputfile:
            print("invalid set")
            QMessageBox.warning(None, "Error", "Input file or output path not set!")
            return
        if self.use_ref_audio and '' == self.selected_ref_audio:
            QMessageBox.warning(None, "Error", "Audio file not set!")
            return

        #if self.get_file_duration(self.inputfile) > 30 and False == self.unlimit_use:
        #    QMessageBox.warning(None, "Error", "File duration > 30s, unsupported in preview version. You can register to have unlimit usage.")
        #    return

        self.startbut.setEnabled(False)
        source_lang = 'english' #self.get_current_source_lang
        target_lang = self.get_current_target_lang()
        role = self.get_current_role()
        speaker_num = self.get_current_speaker_num()
        mod = 0
        if self.use_ref_audio:
            role = self.selected_ref_audio
            mod = 1

        self.translate_process_bar.setValue(0)
        self.translate_process_bar.show()
        self.transthread = TranslatorThread(mod, self.inputfile, source_lang, target_lang, role, self.outputfilePath, speaker_num)
        self.transthread.signal_progress_update.connect(self.update_progress)
        self.transthread.signal_timer.connect(self.update_progress_timer)
        self.transthread.start()
        #
    def update_progress(self, values):
        print("to update val:", values)
        self.translate_process_bar.setValue(values[0])
    def update_progress_timer(self):
        self.transthread.finish()
        self.startbut.setEnabled(True)


