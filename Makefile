IMAGE_NAME := mlflow
CONTAINER := mlflow_server

build:
	docker build -t ${IMAGE_NAME} .

server:
	docker run -d --rm -p 8080:8080 --name ${CONTAINER} ${IMAGE_NAME}

dev:
	docker exec -it ${CONTAINER} /bin/bash

stop:
	docker stop ${CONTAINER}
