import kagglehub

kagglehub.login()
# Download latest version
path = kagglehub.model_download("google/paligemma-2/transformers/paligemma2-3b-pt-224")

print(path)