import gevent
from locust import FastHttpUser, task
from locust.env import Environment
from locust.stats import stats_printer, stats_history, print_percentile_stats
from locust.log import setup_logging
import random
import numpy as np

setup_logging("INFO", None)

class ApiUser(FastHttpUser):
    host = ""

    def on_start(self):
        input = np.array([[3, 2, 4, 0.2], [  4.7, 3, 1.3, 0.2 ]])
        inference_request = {"data": input.tolist() }
        self.requests = inference_request

    @task
    def predict(self):
        self.client.post("/iris/predict", json=self.requests)
        # r = self.client.post("/iris/predict", json=self.requests)
        # print(r.json())

if __name__ == "__main__":
    host = 'http://localhost:30909/models'
    user_count = 16
    spawn_rate = 2
    duration = 60

    # setup Environment and Runner
    env = Environment(host=host, user_classes=[ApiUser])
    env.create_local_runner()

    # start a WebUI instance
    env.create_web_ui("127.0.0.1", 8089)

    # start a greenlet that periodically outputs the current stats
    gevent.spawn(stats_printer(env.stats))

    # start a greenlet that save current stats to history
    gevent.spawn(stats_history, env.runner)

    # start the test

    env.runner.start(user_count, spawn_rate=spawn_rate)

    # in 10 seconds stop the runner
    gevent.spawn_later(duration, lambda: env.runner.quit())

    # wait for the greenlets
    env.runner.greenlet.join()

    print_percentile_stats(env.stats)

    # stop the web server for good measures
    env.web_ui.stop()
