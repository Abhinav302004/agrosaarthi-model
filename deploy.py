# import logging
# from truefoundry.deploy import (
#     Build,
#     LocalSource,
#     Port,
#     Service,
#     NodeSelector,
#     DockerFileBuild,
#     Resources,
# )

# logging.basicConfig(level=logging.INFO)

# service = Service(
#     name="agrosaarthi-backend-new",
#     image=Build(
#         build_source=LocalSource(),
#         build_spec=DockerFileBuild(
#             dockerfile_path="./Dockerfile",
#             build_context_path="./",
#             command="uvicorn app:app --host 0.0.0.0 --port 8000",
#         ),
#     ),
#     resources=Resources(
#         cpu_request=0.5,
#         cpu_limit=0.5,
#         memory_request=1000,
#         memory_limit=1000,
#         ephemeral_storage_request=500,
#         ephemeral_storage_limit=500,
#         node=NodeSelector(),
#     ),
#     env={"MODEL_DIR": "/app/model_files"},
#     ports=[
#         Port(
#             port=8000,
#             protocol="TCP",
#             expose=True,
#             app_protocol="http",
#             host="agrosaarthi-model-ws-18-8000.ml.iit-ropar.truefoundry.cloud",
#         )
#     ],
#     replicas=1.0,
#     auto_shutdown={"idle_timeout": 1800},  # 15 mins idle shutdown

# )


# service.deploy(workspace_fqn="tfy-iitr-az:ws-18", wait=False)
