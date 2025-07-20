from truefoundry.ml import get_client

# Initialize client
client = get_client()

# Download Model Artifact
model_artifact_version = client.get_artifact_version_by_fqn("model:iit-ropar/yolo-best-model/Final_Model:1")
model_download_path = model_artifact_version.download(path="D:/ANNAM.ai/expendables/AgroSaarthiApp/backend/model_files")
print(f"Model downloaded to: {model_download_path}")

# Download Mapping Artifact
mapping_artifact_version = client.get_artifact_version_by_fqn("artifact:iit-ropar/yolo-best-model/Pest_Pesticides_Mapping:1")
mapping_download_path = mapping_artifact_version.download(path="D:/ANNAM.ai/expendables/AgroSaarthiApp/backend/model_files")
print(f"Mapping downloaded to: {mapping_download_path}")