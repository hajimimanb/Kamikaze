# -*- coding: utf-8 -*-
"""面板一体进程：render 循环（5s 写 HTML）+ http.server 8090 线程（serve checkpoints/）。
满足"3 进程架构"：训练 / 评估 / 面板各一个 python 进程。
"""
import os, sys, threading, time

sys.path.insert(0, "C:/agentwork/tools")
from rl_train_dashboard import render

HOST, PORT = "0.0.0.0", 8090
DIR = "C:/agentwork/checkpoints"


def serve():
    import http.server, functools
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DIR)
    httpd = http.server.ThreadingHTTPServer((HOST, PORT), handler)
    print("[panel] http serving %s:%d -> %s" % (HOST, PORT, DIR), flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=serve, daemon=True).start()
    print("[panel] render loop started", flush=True)
    while True:
        try:
            render()
        except Exception as e:
            print("[panel] render err: %s" % e, flush=True)
        time.sleep(5)
