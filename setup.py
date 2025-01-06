import kagglehub

kagglehub.login()
# Download latest version
path = kagglehub.model_download("victorumesiobi/physionet/transformers/1")

print(path)