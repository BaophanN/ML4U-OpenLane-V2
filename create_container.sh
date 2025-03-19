docker run --name lanesegnet\
                --gpus all\
                --mount type=bind,source="/home/baogp4/Bao",target="/workspace/source"\
                --mount type=bind,source="/home/baogp4/datasets",target="/workspace/datasets"\
                --shm-size=16GB\
                -it anhnguyen0912a/lanesegnet:latest
                