import bentoml
from models.experimental import attempt_load
from utils.torch_utils import select_device

model_name = "yolov7-e6e.pt"

device = select_device('cpu')
model = attempt_load(f'./{model_name}', map_location=device)
bentoml.pytorch.save_model('model', model)