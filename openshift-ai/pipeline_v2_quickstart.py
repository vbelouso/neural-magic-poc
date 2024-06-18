from kfp import dsl
from kfp import compiler
from kfp import kubernetes

BASE_MODEL_DIR = "/mnt/models/llm"


@dsl.component(base_image='registry.access.redhat.com/ubi9/python-311',
               packages_to_install=["huggingface-hub"]
               )
def download_model(model_name: str, destination_path: str):
    import subprocess

    # Execute the huggingface_hub-cli command
    result = subprocess.run(["huggingface-cli", "download", model_name,
                             "--local-dir", destination_path,
                             "--local-dir-use-symlinks", "False"], capture_output=True, text=True)

    # Check for errors or output
    if result.returncode == 0:
        print("Model downloaded successfully.")
    else:
        print("Error downloading model:")
        print(result.stderr)


@dsl.component(base_image='quay.io/ltomasbo/sparseml', packages_to_install=["datasets"])
def sparse_model(model_path: str, compress_model_path: str, ds: str, precision: str):
    from sparseml.transformers import (
        SparseAutoModelForCausalLM, SparseAutoTokenizer, load_dataset, oneshot
    )

    model = SparseAutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto")

    tokenizer = SparseAutoTokenizer.from_pretrained(model_path)
    # tokenizer = SparseAutoTokenizer.from_pretrained(model_path).to(model.device)

    dataset = load_dataset(ds)

    def format_data(data):
        return {
            "text": data["instruction"] + data["output"]
        }

    dataset = dataset.map(format_data)

    recipe = """
    test_stage:
      obcq_modifiers:
        LogarithmicEqualizationModifier:
          mappings: [
            [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
            [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
          ]
        QuantizationModifier:
          ignore:
            # These operations don't make sense to quantize
            - LlamaRotaryEmbedding
            - LlamaRMSNorm
            - SiLUActivation
            - MatMulOutput_QK
            - MatMulOutput_PV
            # Skip quantizing the layers with the most sensitive activations
            - model.layers.21.mlp.down_proj
            - model.layers.7.mlp.down_proj
            - model.layers.2.mlp.down_proj
            - model.layers.8.self_attn.q_proj
            - model.layers.8.self_attn.k_proj
          post_oneshot_calibration: true
          scheme_overrides:
            # Enable channelwise quantization for better accuracy
            Linear:
              weights:
                num_bits: 8
                symmetric: true
                strategy: channel
            MatMulLeftInput_QK:
              input_activations:
                num_bits: 8
                symmetric: true
            # For the embeddings, only weight-quantization makes sense
            Embedding:
              input_activations: null
              weights:
                num_bits: 8
                symmetric: false
        SparseGPTModifier:
          sparsity: 0.5
          block_size: 128
          sequential_update: false
          quantize: true
          percdamp: 0.01
          mask_structure: "0:0"
          targets: ["re:model.layers.\\\d*$"]
    """

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        output_dir=compress_model_path,
    )


@dsl.component(base_image='quay.io/ltomasbo/sparseml', packages_to_install=[])
def export_model(model_path: str, exported_model_path: str):
    from sparseml import export

    export(
        model_path,
        task="text-generation",
        sequence_length=1024,
        target_path=exported_model_path
    )


@dsl.component(base_image='quay.io/ltomasbo/sparseml:eval2', packages_to_install=["datasets"],)
def eval_model(model_path: str, tasks: str, batch_size: str):
    import subprocess
    import os

    model_args = "pretrained=" + model_path  # + ",trust_remote_code=True"

    # Execute the huggingface_hub-cli command
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(["python", "./lm-evaluation-harness/main.py",
                             "--model", "sparseml",
                             "--model_args", model_args,
                             "--tasks", tasks,
                             "--batch_size", batch_size,
                             "--no_cache",
                             "--write_out",
                             "--device", "cuda:0",
                             "--num_fewshot", "0",
                             "--limit", "1000"],
                            capture_output=True, text=True, env=env)

    # Check for errors or output
    if result.returncode == 0:
        print("Model evaluated successfully:")
        print(result.stdout)
    else:
        print("Error evaluating the model:")
        print(result.stderr)


@dsl.component(base_image='registry.access.redhat.com/ubi9/python-311', packages_to_install=["boto3"])
def upload_pruned_model(model_path: str):
    import os
    from boto3 import client

    print('Commencing results upload.')
    print(os.environ)

    s3_endpoint_url = os.environ["s3_host"]
    s3_access_key = os.environ["s3_access_key"]
    s3_secret_key = os.environ["s3_secret_access_key"]
    s3_bucket_name = os.environ["s3_bucket"]

    print(f'Uploading predictions to bucket {s3_bucket_name} '
          f'to S3 storage at {s3_endpoint_url}')

    s3_client = client(
        's3', endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key, verify=False
    )

    # Walk through the local folder and upload files
    for root, dirs, files in os.walk(model_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_file_path = os.path.join(
                s3_bucket_name, local_file_path[len(model_path)+1:])
            s3_client.upload_file(
                local_file_path, s3_bucket_name, s3_file_path)
            print(f'Uploaded {local_file_path}')

    print('Finished uploading results.')


@dsl.pipeline(
    name="LLM Pruning Pipeline",
    description="A Pipeline for pruning LLMs with SparseML"
)
def sparseml_pipeline(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    accuracy: int = 90,
    sparse: bool = True,
    eval: bool = False,
    eval_task: str = "hellaswag",
    eval_batch_size: str = "64",
):
    print("Params", model_name, accuracy)
    data_connection = "aws-connection-models"
    volume_name = "models-shared"
    # Directories
    BASE_DIR = "/mnt/models/"
    MODEL_DIR = BASE_DIR + "llm"
    SPARSE_MODEL_DIR = BASE_DIR + "sparse-llm"
    EXPORTED_MODEL_DIR = BASE_DIR + "exported"
    GPU_TYPE = "nvidia.com/gpu"

    # Download volumes
    download_llm_task = download_model(
        model_name=model_name, destination_path=MODEL_DIR)
    kubernetes.mount_pvc(
        download_llm_task,
        pvc_name=volume_name,
        mount_path=BASE_DIR,  # '/mnt/models',
    )
    sparse_llm = None
    export_llm = None
    upload_pruned_llm = None

    # Condition for sparse operation
    with dsl.If(sparse == True, name="sparse=True"):
        sparse_llm = sparse_model(model_path=MODEL_DIR,
                                  compress_model_path=SPARSE_MODEL_DIR,
                                  # ds="open-platypus",
                                  ds="garage-bAInd/Open-Platypus",
                                  precision="bfloat16")
        kubernetes.mount_pvc(
            sparse_llm,
            pvc_name=volume_name,
            mount_path=BASE_DIR,  # '/mnt/models',
        )
        kubernetes.add_toleration(sparse_llm,
                                  key=GPU_TYPE,
                                  operator='Exists',
                                  effect='NoSchedule')
        sparse_llm.set_accelerator_type(GPU_TYPE)
        sparse_llm.set_accelerator_limit(1)
        sparse_llm.after(download_llm_task)

        # Condition for eval operation
        with dsl.If(eval == True, name="eval=True"):
            eval_llm = eval_model(model_path=SPARSE_MODEL_DIR,
                                  tasks=eval_task, batch_size=eval_batch_size)
            kubernetes.mount_pvc(
                eval_llm,
                pvc_name=volume_name,
                mount_path=BASE_DIR,  # '/mnt/models',
            )
            kubernetes.add_toleration(eval_llm,
                                      key=GPU_TYPE,
                                      operator='Exists',
                                      effect='NoSchedule')
            eval_llm.set_accelerator_type(GPU_TYPE)
            eval_llm.set_accelerator_limit(1)
            eval_llm.after(sparse_llm)

            eval_llm_base = eval_model(
                model_path=MODEL_DIR, tasks=eval_task, batch_size=eval_batch_size)
            kubernetes.mount_pvc(
                eval_llm_base,
                pvc_name=volume_name,
                mount_path=BASE_DIR,  # '/mnt/models',
            )
            kubernetes.add_toleration(eval_llm_base,
                                      key=GPU_TYPE,
                                      operator='Exists',
                                      effect='NoSchedule')
            eval_llm_base.set_accelerator_type(GPU_TYPE)
            eval_llm_base.set_accelerator_limit(1)
            eval_llm_base.after(download_llm_task)

            export_llm = export_model(
                model_path=SPARSE_MODEL_DIR, exported_model_path=EXPORTED_MODEL_DIR)
            kubernetes.mount_pvc(
                export_llm,
                pvc_name=volume_name,
                mount_path=BASE_DIR,  # '/mnt/models',
            )
            export_llm.after(eval_llm)

            upload_pruned_llm = upload_pruned_model(
                model_path=EXPORTED_MODEL_DIR)
            kubernetes.use_secret_as_env(upload_pruned_llm,
                                         secret_name=data_connection,
                                         secret_key_to_env={'AWS_ACCESS_KEY_ID': 's3_access_key',
                                                            'AWS_SECRET_ACCESS_KEY': 's3_secret_access_key',
                                                            'AWS_S3_ENDPOINT': 's3_host',
                                                            'AWS_S3_BUCKET': 's3_bucket'})
            kubernetes.mount_pvc(
                upload_pruned_llm,
                pvc_name=volume_name,
                mount_path=BASE_DIR,  # '/mnt/models',
            )
            upload_pruned_llm.after(export_llm)

        with dsl.If(eval == False, name="eval=False"):
            export_llm = export_model(
                model_path=SPARSE_MODEL_DIR, exported_model_path=EXPORTED_MODEL_DIR)
            kubernetes.mount_pvc(
                export_llm,
                pvc_name=volume_name,
                mount_path=BASE_DIR,  # '/mnt/models',
            )
            export_llm.after(sparse_llm)

            upload_pruned_llm = upload_pruned_model(
                model_path=EXPORTED_MODEL_DIR)
            kubernetes.use_secret_as_env(upload_pruned_llm,
                                         secret_name=data_connection,
                                         secret_key_to_env={'AWS_ACCESS_KEY_ID': 's3_access_key',
                                                            'AWS_SECRET_ACCESS_KEY': 's3_secret_access_key',
                                                            'AWS_S3_ENDPOINT': 's3_host',
                                                            'AWS_S3_BUCKET': 's3_bucket'})
            kubernetes.mount_pvc(
                upload_pruned_llm,
                pvc_name=volume_name,
                mount_path=BASE_DIR,  # '/mnt/models',
            )
            upload_pruned_llm.after(export_llm)
    with dsl.If(sparse == False, name="sparse=False"):
        with dsl.If(eval == True, name="eval=True"):
            eval_llm_base = eval_model(
                model_path=MODEL_DIR, tasks=eval_task, batch_size=eval_batch_size)
            kubernetes.mount_pvc(
                eval_llm_base,
                pvc_name=volume_name,
                mount_path=BASE_DIR,  # '/mnt/models',
            )
            kubernetes.add_toleration(eval_llm_base,
                                      key=GPU_TYPE,
                                      operator='Exists',
                                      effect='NoSchedule')
            eval_llm_base.set_accelerator_type(GPU_TYPE)
            eval_llm_base.set_accelerator_limit(1)
            eval_llm_base.after(download_llm_task)

        export_llm = export_model(model_path=MODEL_DIR,
                                  exported_model_path=EXPORTED_MODEL_DIR)
        kubernetes.mount_pvc(
            export_llm,
            pvc_name=volume_name,
            mount_path=BASE_DIR,  # '/mnt/models',
        )
        export_llm.after(download_llm_task)
        upload_pruned_llm = upload_pruned_model(model_path=EXPORTED_MODEL_DIR)
        kubernetes.use_secret_as_env(upload_pruned_llm,
                                     secret_name=data_connection,
                                     secret_key_to_env={'AWS_ACCESS_KEY_ID': 's3_access_key',
                                                        'AWS_SECRET_ACCESS_KEY': 's3_secret_access_key',
                                                        'AWS_S3_ENDPOINT': 's3_host',
                                                        'AWS_S3_BUCKET': 's3_bucket'})
        kubernetes.mount_pvc(
            upload_pruned_llm,
            pvc_name=volume_name,
            mount_path=BASE_DIR,  # '/mnt/models',
        )
        upload_pruned_llm.after(export_llm)


compiler.Compiler().compile(sparseml_pipeline,
                            package_path='pipeline_v2_quickstart.yaml')
