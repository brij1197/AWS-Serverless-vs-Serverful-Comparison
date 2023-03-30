import os

os.system("bentoml delete -y pytorch_model_aws:yolo_aws")
print("---------Deleted bentos = pytorch_model_aws:yolo_aws---------\n")

os.system("bentoml build --version yolo_aws")
print("---------Built new bentos---------\n")

# Uncomment below if deployment_config.yaml
# terraform init

os.system("bentoctl build -b pytorch_model_aws:yolo_aws -f deployment_config.yaml")
print("---------Built bentoctl package---------\n")

os.system("terraform apply -var-file bentoctl.tfvars --auto-approve")
print("---------Applied Terraform---------")

# bentoctl destroy -f deployment_config.yaml --auto-approve
