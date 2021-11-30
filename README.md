# mlserver-bench

## run bench
0. install python requirements
    - pip install -r requirements.txt

1. generate dummy model file
    - python generate_dummy_model.py

2. start the server
    - docker-compose up

3. run load test 
    - python test_performance.py