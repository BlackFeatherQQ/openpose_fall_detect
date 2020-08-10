#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhaoyuxin
@connect: zhaoyx77@163.com
@file: thinter_s.py
@time: 2020/8/4 20:02
'''

import tkinter as tk
from tkinter import  messagebox
import sys
import os
import threading
import fileinput
from demo import detect_main


address_dir = 'Video_Information/url_address.txt'
name_dir = 'Video_Information/id_name.txt'

def add():
    var_id = id_name.get()
    t_name.insert('insert', var_id)
    t_name.insert('end', '\n')
    b = 'Video_Information/id_name.txt'
    h = open(b, 'a+', encoding='utf-8')
    h.write(var_id + '\n')
    h.close()

    var_adress = url_address.get()
    t_address.insert('insert', var_adress)
    t_address.insert('end', '\n')
    c = 'Video_Information/url_address.txt'
    f = open(c, 'a+', encoding='utf-8')
    f.write(var_adress + '\n')
    f.close()

def stop():
    sys.exit()

def test(txt):
    detect_main(video_source=txt['url_address'],video_name=txt['id_name'])

def run():
    adress_txt = open(address_dir, 'r', encoding='utf-8')
    name_txt = open(name_dir, 'r', encoding='utf-8')
    for names in zip(name_txt,adress_txt):
        key = names[0].strip()
        value = names[1].strip()
        txts = {'id_name': key, 'url_address': value}
        thread_ = threading.Thread(target=test,args=(txts,))
        thread_.setDaemon(True)
        thread_.start()

def delete():
    delete_name = delete_id_name.get()
    line_num = 0
    with open('Video_Information/id_name.txt', 'r') as a:
        is_exist = False
        for i,id_names in enumerate(a):
            id_names = id_names.strip()
            if id_names == delete_name:
                is_exist = True
                line_num = i + 1
        if not is_exist:
            tk.messagebox.showerror('Error', 'Your id name is wrong,try it again!')
        a.close()

        f = fileinput.input('Video_Information/url_address.txt', inplace=1, mode='rU')
        for line in f:
            if f.filelineno() == line_num:
                print(end='')
            else:
                print(line,end='')
        f.close()

        g = fileinput.input('Video_Information/id_name.txt', inplace=1, mode='rU' )
        for line in g:
            if g.filelineno() == line_num:
                print(end='')
            else:
                print(line,end='')
        g.close()

        if os.path.exists(name_dir):
            a = open(name_dir, 'r', encoding='utf-8')
            t_name.delete(0.0, index2='end')
            for id_names in a:
                t_name.insert('insert', id_names)
            a.close()

            b = open(address_dir, 'r', encoding='utf-8')
            t_address.delete(0.0, index2='end')
            for id_names in b:
                t_address.insert('insert', id_names)
            b.close()

if __name__ == '__main__':

    # prosess_pool = Pool(processes=2)
    #
    # parser = argparse.ArgumentParser(
    #     description='''Lightweight human pose estimation python demo.
    #                            This is just for quick results preview.
    #                            Please, consider c++ demo for the best performance.''')
    # parser.add_argument('--checkpoint-path', type=str, default='openpose.jit',
    #                     help='path to the checkpoint')
    # parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    # parser.add_argument('--prosess-pool', type=Pool, default=None)
    #
    # args = parser.parse_args()
    # args.prosess_pool = prosess_pool

    if not os.path.exists('Video_Information'):
        os.makedirs('Video_Information')

    window = tk.Tk()
    window.title('Passenger fall alarm system')
    window.geometry('550x500')
    # window.iconbitmap('px.ico')
    tk.Label(window, text='please input id name:').place(x=10, y=10)
    tk.Label(window, text='please input url address:').place(x=10, y=40)
    tk.Label(window, text='please input delete id name:').place(x=10, y=70)
    tk.Label(window, text='id names').place(x=20, y=160)
    tk.Label(window, text='url addresses').place(x=250, y=160)

    var_id_name = tk.StringVar()
    id_name = tk.Entry(window, textvariable=var_id_name)
    id_name.place(x=200, y=10)

    var_url_address = tk.StringVar()
    url_address = tk.Entry(window, textvariable=var_url_address)
    url_address.place(x=200, y=40)

    var_delete_name = tk.StringVar()
    delete_id_name = tk.Entry(window, textvariable=var_delete_name)
    delete_id_name.place(x=200, y=70)

    add = tk.Button(window, text='add', width=10, command=add)
    add.place(x=70, y=120)

    stop = tk.Button(window, text='stop', width=10, command=stop, background='red')
    stop.place(x=315, y=120)

    run = tk.Button(window, text='run', width=10, command=run)
    run.place(x=195, y=120)

    t_name = tk.Text(window, width=10, height=15, font=14)
    t_name.place(x=20, y=180)

    t_address = tk.Text(window, width=50, height=15, font=14)
    t_address.place(x=100, y=180)

    if os.path.exists(name_dir):
        a = open(name_dir, 'r', encoding='utf-8')
        for id_names in a:
            t_name.insert('insert', id_names)
        a.close()

        b = open(address_dir, 'r', encoding='utf-8')
        for id_names in b:
            t_address.insert('insert', id_names)
        b.close()

    delete = tk.Button(window, text='delete', width=10, command=delete)
    delete.place(x=430, y=120)

    window.mainloop()
