#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
ctime = 2017/02/17
"""
import run_couplet as p2w
from run_couplet import create_model,decode_once2
import tensorflow as tf
from flask import Flask,request
from log_util import *
app = Flask(__name__)
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class Singleton(object):
    # 定义静态变量实例
    __singleton = None
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        if Singleton.__singleton is None:
            sess = tf.Session()
            Singleton.__singleton = sess, create_model(sess, True)
        return Singleton.__singleton

@app.route("/")
def hello():
    import time
    s = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    return "Hello World!\n%s"%(s)

@app.route('/couplet')
def couplet():
    key = 'text'
    val = request.args.get(key)

    sess, model = Singleton.get_instance()
    ret = decode_once2(sess, model, val)
    ret = ret.replace(" ","")
    log.info("input%s,output:%s"%(val, ret))
    ret = "<h1>key:%s <br> value:%s</h1>"%(val, ret)
    return ret

@app.route('/couplet2')
def couplet2():
    key = 'text'
    val = request.args.get(key)
    sess, model = Singleton.get_instance()
    ret = decode_once2(sess, model, val)
    ret = ret.replace(" ","")
    # print("input:%s,output:%s"%(val,ret))
    # ret = "<h1>input:%s <br> output:%s</h1>"%(val, ret)
    log.info("input%s,output:%s"%(val, ret))
    return ret


if __name__ == "__main__":
    # app.run(host='172.18.8.181', port=5001)
    app.run(host='172.18.4.193', port=5002)
    # app.run(port=5002)

    # val = "nihao"
    # sess, model = Singleton.get_instance()
    # ret = decode_once2(sess, model, val)
    # print("ret", ret)
    #
    # val = "nihaoma"
    # sess, model = Singleton.get_instance()
    # ret = decode_once2(sess, model, val)
    # print("ret", ret)
