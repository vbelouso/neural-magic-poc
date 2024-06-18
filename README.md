# How to deploy the pipeline and/or serve the model

The pipeline is to make use of [SparseML](https://github.com/neuralmagic/sparseml) to optimize the model, and then
the [KServe](https://kserve.github.io/website/latest) InferenceService/ServingRuntime are the one running the
[DeepSparse](https://github.com/neuralmagic/deepsparse) runtime with the model

## Prerequisites

Created data science project in Red Hat OpenShift AI

## Create object data store (MinIO) for the models

Create namespace for the object store if you don't have one

```bash
oc new-project object-datastore
```

Deploy MinIO:

```bash
oc apply -f minio.yaml
```

Create a couple of buckets in MinIO using credentials from the created `minio-secret`

- one for the pipeline (e.g., named `mlops`)
- one for the models (e.g., named `models`).

## SparseML

### Pipeline server

Create [pipeline server](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/2.8/html/working_on_data_science_projects/working-with-data-science-pipelines_ds-pipelines#configuring-a-pipeline-server_ds-pipelines), pointing to an S3 bucket
> [!TIP]
> For the `Access key` and `Secret key` use the credentials from the `minio-secret` (Access key=minio_root_user, Secret key=minio_root_password)
>
> For the `Endpoint` use `http://minio-service.object-datastore.svc.cluster.local:9000`
>
> For the `Bucket` use `mlops`

**For RHOAI < 2.9**

[Import](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/2.8/html/working_on_data_science_projects/working-with-data-science-pipelines_ds-pipelines#importing-a-data-science-pipeline_ds-pipelines) the existing PipelineRun [sparseml_pipeline.yaml](openshift-ai/sparseml_pipeline.yaml) into the Red Hat OpenShift AI or generate a new one via the commands:

> [!IMPORTANT]
> File `sparseml_pipeline_custom.yaml` should be created as a result of executing the command

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install kfp kfp_tekton==1.5.9
python openshift-ai/pipeline.py
```

**For RHOAI >= 2.9**

[Import](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/2.9/html/working_on_data_science_projects/working-with-data-science-pipelines_ds-pipelines#importing-a-data-science-pipeline_ds-pipelines) the existing PipelineRun [pipeline_v2_quickstart.yaml](openshift-ai/pipeline_v2_quickstart.yaml) into the Red Hat OpenShift AI or generate a new one via the commands:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install kfp==2.8.0
pip install kfp-kubernetes==1.2.0
python openshift-ai/pipeline_v2_quickstart.py
```

> [!NOTE]
> if some of the steps may take longer than one hour you either need to change the defaults for taskRuns in Red Hat OpenShift AI or add a timeout: Xh per taskRun.
  You can see `sparseml_simplified_pipeline.yaml` and search for `timeout: 5h` to see an example.

### Pipeline Requirements

Cluster storage (created via PersistentVolumeClaims) named `models-shared`, so that a volume to be shared is created

Data connection, named `models`, pointing to the S3 bucket to store the resulting model

> [!NOTE]
> NOTE: the cluster storage and the data connection can have any name, as long as it is the same given later on the pipeline parameters.

### Create the images needed for the pipeline

Build the container images for the sparsification and the evaluation steps

```bash
USER="<your_username>"
podman build -t quay.io/${USER}/neural-magic:sparseml -f openshift-ai/sparseml_Dockerfile .
podman build -t quay.io/${USER}/neural-magic:sparseml_eval -f openshift-ai/sparseml_eval_Dockerfile .
podman build -t quay.io/${USER}/neural-magic:nm_vllm_eval -f openshift-ai/nm_vllm_eval_Dockerfile .
podman build -t quay.io/${USER}/neural-magic:base_eval -f openshift-ai/base_eval_Dockerfile .
```

Push the container images to a registry

```bash
podman push quay.io/${USER}/neural-magic:sparseml
podman push quay.io/${USER}/neural-magic:sparseml_eval
podman push quay.io/${USER}/neural-magic:nm_vllm_eval
podman push quay.io/${USER}/neural-magic:base_eval
```

> [!WARNING]
> Haven't tested the following section, please jump to [Run the pipeline](#run-the-pipeline)

### (OLD) Compile the pipeline (RHOAI >= 2.9)

This is the process to create the `PipelineRun` yaml file from the python script. It requires `kfp_tekton` version 1.5.9:

```bash
pip install kfp_tekton==1.5.9
python pipeline_simplified.py
```

- NOTE: there is another option for a more complex/flexible pipeline at `pipeline_nmvllm.py`, but the rest assumes the usage of the simplified one.

### (NEW) Compile the pipeline for V2 (RHOAI >= 2.9)

This is the process to create the pipeline `yaml` file from the python script.
It requires `kfp.kubernetes`:

```bash
pip install kfp[kubernetes]
python pipeline_v2_cpu.py
python pipeline_v2_gpu.py
```

- NOTE: there are two different pipelines for V2, one for GPU and one for CPU.
  It would be straightforward to merge them in one and have a pipeline parameter
  to chose between them

### Run the pipeline

Run the pipeline selecting the model and the options:

- Evaluate or not
- GPU (Quantized) or CPU (Sparsified: Quantized + Pruned). Note for GPU inferencing, it is not supported to both prune and quantized yet.

## DeepSparse

Run the optimized model with DeepSparse

### Create the image needed for the Inference Service

Build a container image

```bash
podman build -t quay.io/${USER}/neural-magic:deepsparse -f deepsparse_Dockerfile .
```

Push the container image to a registry

```bash
podman push quay.io/${USER}/neural-magic:deepsparse
```

### Option A: Deploy through ServingRuntime

Note DeepSparse require write access to the mounted volume with the model, so doing a workaround so that it gets copied to an extra mount with `ReadOnly` set to `False`.

If you have created **a custom image**, you need to update the container image in the specified file.

```bash
oc apply -f openshift-ai/serving_runtime_deepsparse.yaml
```

And them from the Red Hat OpenShift AI you can deploy a model using it and pointing to the `models` DataConnection

### Option B: Deploy InferenceService

Create a secret and a Service Account that points to the S3 endpoint. Modified them as needed.

```bash
oc apply -f openshift-ai/secret.yaml
oc apply -f openshift-ai/sa.yaml

oc apply -f openshift-ai/inference.yaml
```

## nm-vLLM

Run the optimized model with nm-vLLM

### Create the image needed for the ServingRuntime

Build the container with:

```bash
podman build -t quay.io/USER/neural-magic:nm-vllm -f nmvllm_Dockerfile .
```

And push it to a registry

```bash
podman push quay.io/USER/neural-magic:nm-vllm
```

### Deploy through ServingRuntime

Note DeepSparse require write access to the mounted volume with the model, so doing a workaround so that it gets copied to an extra mount with `ReadOnly` set to `False`.

```bash
oc apply -f openshift-ai/serving_runtime_vllm.yaml
oc apply -f openshift-ai/serving_runtime_vllm_marlin.yaml
```

And them from the Red Hat OpenShift AI you can deploy a model using it and pointing to the `models` DataConnection. You can use one or the other depending on running sparsified models or quantized (with `marlin`) models.

## Testing with Gradio

Run the request.py and access the Gradio server deployed locally at `127.0.0.1:7860`. Update the URL with the one from the deployed runtime (`ksvc` route)

```bash
python openshift-ai/request.py
```
