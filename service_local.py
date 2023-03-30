import bentoml
import boto3
import botocore
import os
import time
import cv2
import torch
from bentoml.io import JSON
from pathlib import Path
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

ACCESS_KEY = ""
SECRET_KEY = ""
BUCKET_NAME = ""
FINAL_BUCKET_NAME = ""


async def _predict(model_name, model_runner, img_src, img_size):
    device = select_device('cpu')
    model = attempt_load(f'./{model_name}', map_location=device)
    stride = 64
    imgsz = check_img_size(img_size, s=stride)
    dataset = LoadImages(img_src, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    save_dir = Path(increment_path(Path("runs/cmpt-756-results") / 'exp', exist_ok=False))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time_synchronized()
        with torch.no_grad():
            pred = (await model_runner.async_run(img))[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
        t3 = time_synchronized()
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    return save_path


def s3_download(BUCKET_NAME, KEY):
    session = boto3.session.Session(aws_access_key_id=ACCESS_KEY,
                                    aws_secret_access_key=SECRET_KEY)
    try:
        session.client('s3').download_file(BUCKET_NAME, KEY, KEY)
        print("Downloaded")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")


def s3_image_upload(SAVE, BUCKET_NAME, KEY):
    session = boto3.session.Session(aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    try:
        session.client('s3').upload_file(SAVE, BUCKET_NAME, KEY)
        print("Uploaded")
    except botocore.exceptions.ClientError as e:
        print(e)


model_runner = bentoml.pytorch.get("model:latest").to_runner()
svc = bentoml.Service("pytorch_model", runners=[model_runner])


@svc.api(input=JSON(), output=JSON())
async def predict(parsed_json: JSON) -> JSON:
    KEY = parsed_json["filename"]
    s3_download(BUCKET_NAME, KEY)
    SAVE = await _predict(model_name="yolov7.pt", model_runner=model_runner, img_src=KEY, img_size=640)
    s3_image_upload(SAVE, FINAL_BUCKET_NAME, KEY)
    return {"location": f"https://{FINAL_BUCKET_NAME}.s3.amazonaws.com/{KEY}"}
