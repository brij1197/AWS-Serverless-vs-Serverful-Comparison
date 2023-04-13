import bentoml
import boto3
import botocore
import os
import time
import json
import cv2
import torch
from pprint import pprint
from bentoml.io import JSON
from pathlib import Path
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from uuid import uuid4


async def _predict(model_name: Path, model_runner, img_src: Path, img_size: int):
    device = select_device('cpu')
    model = attempt_load(model_name, map_location=device)
    stride = 64
    imgsz = check_img_size(img_size, s=stride)
    dataset = LoadImages(img_src, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
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
        pred = non_max_suppression(
            pred, 0.25, 0.45, classes=None, agnostic=False)
        t3 = time_synchronized()
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = Path(img_src.parent).joinpath(
                img_src.stem + '_output' + img_src.suffix)
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label,
                                 color=colors[int(cls)], line_thickness=1)
            if dataset.mode == 'image':
                cv2.imwrite(save_path.as_posix(), im0)
    return True, save_path


async def s3_download(s3_obj, bucket_name, object_name, use_uuid=True):
    if use_uuid:
        filepath = Path(
            '/tmp/' + str(uuid4()).replace("-", "") + "_" + object_name)
    else:
        filepath = Path(f'/tmp/{object_name}')
    try:
        if not filepath.exists():
            s3_obj.download_file(bucket_name, object_name, filepath)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
    return filepath


async def s3_upload(s3_obj, bucket_name, filepath):
    try:
        s3_obj.upload_file(filepath, bucket_name, filepath.name,
                           ExtraArgs={'ACL': 'public-read'})
    except botocore.exceptions.ClientError as e:
        print(e)


def remove_file(filepath):
    if filepath.exists():
        os.remove(filepath.as_posix())
        return True


model_runner = bentoml.pytorch.get("yolo:latest").to_runner()
svc = bentoml.Service("pytorch_model_aws", runners=[model_runner])


@svc.api(input=JSON(), output=JSON())
async def predict(parsed_json: JSON) -> JSON:
    MODEL_BUCKET_NAME = ''
    BUCKET_NAME = ''
    FINAL_BUCKET_NAME = ''
    YOLO_MODEL_VERSION = 'yolov7.pt'

    try:
        begin_time = time.time()
        timings = {
            "Model_Download_Time_S3": 0, "Image_Download_Time_S3": 0, "Inference_Time": 0,
            "Total_Time": 0, "Image_Upload_Time_S3": 0,
        }
        KEY = parsed_json["filename"]
        s3_obj = boto3.client('s3')
        start_time = time.time()
        model_filpath = await s3_download(s3_obj, MODEL_BUCKET_NAME, YOLO_MODEL_VERSION, use_uuid=False)
        timings['Model_Download_Time_S3'] = f"{time.time() - start_time:.3f}s"

        start_time = time.time()
        img_filepath = await s3_download(s3_obj, BUCKET_NAME, KEY)
        timings['Image_Download_Time_S3'] = f"{time.time() - start_time:.3f}s"

        start_time = time.time()
        outcome, output_filepath = await _predict(
            model_name=model_filpath,
            model_runner=model_runner,
            img_src=img_filepath,
            img_size=640
        )
        timings['Inference_Time'] = f"{time.time() - start_time:.3f}s"

        if outcome:
            start_time = time.time()
            await s3_upload(s3_obj, FINAL_BUCKET_NAME, output_filepath)
            timings['Image_Upload_Time_S3'] = f"{time.time() - start_time:.3f}s"

            timings['Total_Time'] = f"{time.time() - begin_time:.3f}s"
            pprint(timings)
    except Exception as e:
        print(e)
        raise bentoml.exceptions.InternalServerError(str(e))
    finally:
        pprint({"Model_Filepath": model_filpath,
               "Img_Filepath": img_filepath, "Output_Filepath": output_filepath})
        remove_file(img_filepath)
        remove_file(output_filepath)
        print("Deleted all files")

    return {
        "bucket": FINAL_BUCKET_NAME,
        "filename": output_filepath.name,
        "location": f"https://{FINAL_BUCKET_NAME}.s3.amazonaws.com/{output_filepath.name}"
    }
