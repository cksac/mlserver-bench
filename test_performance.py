import gevent
from locust import FastHttpUser, task
from locust.env import Environment
from locust.stats import stats_printer, stats_history, print_percentile_stats
from locust.log import setup_logging
import random

setup_logging("INFO", None)

class ApiUser(FastHttpUser):
    host = "http://localhost:9091/v2/models"

    def on_start(self):
        self.requests = []         
        for i in range(1, 10):
            inference_request = {
                "inputs": [
                    {
                        "name": "predict",
                        "datatype": "BYTES",
                        "shape": [9],
                        "data": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
                        "parameters": {
                            "content_type": "str"
                        }
                    }
                ]
            }
            self.requests.append(inference_request)

    @task
    def predict(self):
        request = random.choice(self.requests)
        self.client.post("/model-a/infer", json=request)
        # r = self.client.post("/model-a/infer", json=request)
        # print(r.content)

if __name__ == "__main__":
    user_count = 4
    spawn_rate = 4
    duration = 60

    # setup Environment and Runner
    env = Environment(user_classes=[ApiUser])
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
