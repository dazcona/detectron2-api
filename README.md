## Detectron2 API

API that receives images, analyses them using Detectron2 and gives images back annotated including masks for all objects detected

Code adapted from:
* [Enric Moreu's Human Mask Generator API](https://github.com/enric1994/detectron-api)
* [Facebook's Detectron2](https://github.com/facebookresearch/detectron2)

## Technologies

* [Flask](http://flask.pocoo.org/)
* [Docker](https://www.docker.com/)

## Deployment

1. From the `docker` folder run: `docker-compose up -d` . You can check the status of the container with: `docker ps -a`. You can also see what's going on inside the container with `docker logs dazcona_express_human-mask-api -f`. Then, either as the command to be run by the container on start-up or manually, you have to start the Flask development server:
```
$ cd docker
$ docker exec -it dazcona_express_human-mask-api bash
# python3 src/app.py
```

2. On the main folder run: `python3 send_file.py` to send one or more images. A list of images will be generated based on the objects detected by Detectron2 including masks and annotations. These images will be stored in `figures/responses/`
```
# python3 send_file.py
```

## Example

### Input

<img src="figures/samples/nike.jpg" width="50%">

### Output

<img src="figures/responses/nike-annotated.jpg" width="50%">

| ![](figures/responses/nike-detection-roi-person-0.9992-209_393_983_1342.jpg) | ![](figures/responses/nike-detection-mask-person-0.9992-209_393_983_1342.jpg) | ![](figures/responses/nike-mask-person-0.9992-209_393_983_1342.jpg) | ![](figures/responses/nike-detection-reverse-mask-person-0.9992-209_393_983_1342.jpg) | ![](figures/responses/nike-reverse-mask-person-0.9992-209_393_983_1342.jpg) |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| ![](figures/responses/nike-detection-roi-person-0.9900-0_261_493_1214.jpg) | ![](figures/responses/nike-detection-mask-person-0.9900-0_261_493_1214.jpg) | ![](figures/responses/nike-mask-person-0.9900-0_261_493_1214.jpg) | ![](figures/responses/nike-detection-reverse-mask-person-0.9900-0_261_493_1214.jpg) | ![](figures/responses/nike-reverse-mask-person-0.9900-0_261_493_1214.jpg) |
| ![](figures/responses/nike-detection-roi-person-0.9966-681_302_1075_1296.jpg) | ![](figures/responses/nike-detection-mask-person-0.9966-681_302_1075_1296.jpg) | ![](figures/responses/nike-mask-person-0.9966-681_302_1075_1296.jpg) | ![](figures/responses/nike-detection-reverse-mask-person-0.9966-681_302_1075_1296.jpg) | ![](figures/responses/nike-reverse-mask-person-0.9966-681_302_1075_1296.jpg) |
| ![](figures/responses/nike-detection-roi-sports_ball-0.9704-169_1108_537_1348.jpg) | ![](figures/responses/nike-detection-mask-sports_ball-0.9704-169_1108_537_1348.jpg) | ![](figures/responses/nike-mask-sports_ball-0.9704-169_1108_537_1348.jpg) | ![](figures/responses/nike-detection-reverse-mask-sports_ball-0.9704-169_1108_537_1348.jpg) | ![](figures/responses/nike-reverse-mask-sports_ball-0.9704-169_1108_537_1348.jpg) |

