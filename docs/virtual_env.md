# Virtual environments for computation

## 1. Python venv 

One can set up a local Python virtual environment. This approach is useful to control over dependencies and execution.

First, create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Instal your Python dependencies. 

Exit the virtual environment: 

```
deactivate
```

## 2. Docker 

Docker installed (version 29.5). 

```
sudo docker run hello-world
```
Installation working. To be tested in more operation. 


Check running dockers: 

```
systemctl status docker

```

Stop docker: 
```
docker ps

docker stop <container_id>
```
