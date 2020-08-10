from tkinter import Tk, ACTIVE, DISABLED
from subprocess import Popen, PIPE
import os
import time
from re import findall
import sys
from tkinter import Label, Button, StringVar,Entry
from tkinter.messagebox import showwarning, askyesno
from demo import detect_main
import threading


os.environ["PYTORCH_JIT"] = "0"

root = Tk()
setWidth, setHeight = root.maxsize()
root.geometry('320x220+%d+%d' % ((setWidth-320)/2, (setHeight)/2-220))
root.title('Operation Assistant')
root.resizable(width=False, height=False)


def open_explo(url):
    Popen('chrome %s' % url)


def find_process():
    proc = Popen('netstat -ano | findstr "8000"', shell=True, stdout=PIPE).stdout.read()
    print(proc)
    return proc


def kill_process(res:str):
    try:
        pid_value = findall(r'LISTENING\s+?(\d+)', res.decode())[0]
    except:
        if "TIME_WAIT" in res.decode():
            showwarning(title='Tips', message='8000 The port is not completely released. Please try again later。')
        else:
            showwarning(title='Tips', message='Error: unknown error')

        root.destroy()
        sys.exit(0)
    Popen('taskkill /F /pid %s' % pid_value, shell=True, stdout=PIPE)


def check_btn():
    if bvar1.get()=="stop":
        button_index.config(state=ACTIVE)
        button_admin.config(state=ACTIVE)
    else:
        button_index.config(state=DISABLED)
        button_admin.config(state=DISABLED)
    root.update()

def showinfo():
    # 获取输入的内容

    print(label1.get())
    detect_main(video_source=label1.get(),video_name='ssbbb')
    bvar1.set('run')
    check_btn()
    switch_btn['background'] = "#EBEDEF"
    time.sleep(0.5)
    bottom_message['text'] = "be ready"

def state_sw():
    if switch_btn['text'] != "stop":
        if label1.get():
            run_shell('python demo.py runserver')
            bvar1.set('stop')
            switch_btn['background'] = "#32A084"
            # showinfo(title='提示信息', message='开始运行')
            bottom_message['text'] = "start running"
            check_btn()
            time.sleep(0.5)
            bottom_message['text'] = "Service started"
            light = threading.Thread(target=showinfo, )
            light.setDaemon(True)
            light.start()
        else:
            showwarning(title='Tips', message='The file does not exist!')
    else:
        if askyesno('Tips', 'this stop key does not work,sorry', default='no'):  #'Do you want to stop the service？'
            search_res = find_process()
            if search_res:
                kill_process(search_res)
                bvar1.set('run')
                bottom_message['text'] = "Out of Service"
                check_btn()
                switch_btn['background'] = "#EBEDEF"
                time.sleep(0.5)
                bottom_message['text'] = "be ready"
            else:
                bottom_message['text'] = "Not ready"
                showwarning(title='Tips', message='The service process does not exist!')
                bvar1.set('run')
                bottom_message['text'] = "be ready"
                check_btn()
                switch_btn['background'] = "#EBEDEF"


def run_shell(run_param):
    mark = time.strftime('RA+%Y%m%d %H:%M:%S', time.localtime()) # 用于进程名称的特征字符串，方便过滤
    cmd = 'start run_assistant.bat "%s" %s' % (mark, run_param)
    console = Popen(cmd, shell=True)
    if run_param == "python demo.py runserver":
        return
    root.withdraw()
    console.wait()
    while True:
        task_info = Popen('tasklist /V | findstr /C:"%s"' % mark, shell=True, stdout=PIPE)
        if not task_info.stdout.read():
            root.deiconify()
            break


bvar1 = StringVar()
bvar1.set('run')

label1 = Entry(root, text='webserver',width=25,borderwidth=2,relief='groove',background='#f60',foreground='white')
switch_btn = Button(root, textvariable=bvar1,background='#EBEDEF',command=state_sw)
label1.grid(row=0,column=0,columnspan=5,padx=15,pady=10,ipadx=5,ipady=6)
switch_btn.grid(row=0,column=5,padx=30,pady=10,ipadx=5,ipady=2)


button_index = Button(root, text='mainpage')
button_index.grid(row=4,column=3,padx=10,ipadx=5,ipady=2)
button_admin = Button(root, text='console')
button_admin.grid(row=4,column=4,ipady=2)

bottom_message = Label(foreground='blue',width=36,anchor='w',font=('Arial', 8))
bottom_message.grid(row=5,column=0,columnspan=6,padx=15,ipadx=5,sticky='W')

ifSetup = find_process()
check_btn()


if __name__ == '__main__':
    root.mainloop()