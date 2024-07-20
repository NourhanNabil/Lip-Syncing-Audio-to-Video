

## Clone the repository
```bash
git clone https://github.com/NourhanNabil/Lip-Syncing-Audio-to-Video.git
cd Lip-Syncing-Audio-to-Video
```

### Build environment

We recommend a python version >=3.10 and cuda version =11.7. Then build environment as follows:

``` bash 
pip install -r requirements.txt
```

### mmlab packages
``` bash 
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```
### Download ffmpeg-static
``` bash
 wget -O ffmpeg-release-amd64-static.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
 tar -xvf ffmpeg-release-amd64-static.tar.xz -C musetalk
```
``` bash
  export FFMPEG_PATH=/musetalk/ffmpeg-7.0.1-amd64-static
``` 


### Prerequisites
- Docker installed on your machine.
- Docker Compose installed on your machine.

## Docker

1. Run the Docker container:
``` bash 
docker-compose up -d
```
2. For interactive shell access while the container is running, you can use:
``` bash 
docker-compose exec lip-syncing-service bash
```
3. Shut Down the Containers:
```bash
docker-compose down # Stops and removes containers, networks, volumes, and other services.
docker-compose stop # Stops containers without removing them, allowing you to start them again later.
```

### Lip-Syncing-Audio-to-Video API
- **Endpoint**: `/upload`
- **Method**: POST
- **Request Body**: 2 File upload for video and audio

### Swagger UI
- Access the API documentation and test the endpoints using the Swagger UI at `http://localhost:8080/docs`. 








