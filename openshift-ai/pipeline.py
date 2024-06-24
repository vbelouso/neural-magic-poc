import kfp.dsl as dsl
import kfp.components as comp
from kfp_tekton.compiler import TektonCompiler
from kfp_tekton.k8s_client_helper import env_from_secret

from kubernetes.client import V1Volume, V1PersistentVolumeClaimVolumeSource, V1Toleration

BASE_MODEL_DIR = "/mnt/models/llm"

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


def sparse_model(model_path:str, compress_model_path: str, ds: str, precision: str):
    from sparseml.transformers import (
        SparseAutoModelForCausalLM, SparseAutoTokenizer, load_dataset, oneshot
    )

    model = SparseAutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    tokenizer = SparseAutoTokenizer.from_pretrained(model_path)
    #tokenizer = SparseAutoTokenizer.from_pretrained(model_path).to(model.device)

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


def export_model(model_path: str, exported_model_path: str):
    from sparseml import export

    export(
        model_path,
        task="text-generation",
        sequence_length=1024,
        target_path=exported_model_path
    )


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
            s3_file_path = os.path.join(s3_bucket_name, local_file_path[len(model_path)+1:])
            s3_client.upload_file(local_file_path, s3_bucket_name, s3_file_path)
            print(f'Uploaded {local_file_path}')

    print('Finished uploading results.')


download_op = comp.create_component_from_func(download_model,
                                              packages_to_install=["huggingface-hub"],
                                              base_image='registry.access.redhat.com/ubi9/python-311')
sparse_op = comp.create_component_from_func(sparse_model,
                                            packages_to_install=["datasets"],
                                            base_image='quay.io/ltomasbo/sparseml')
export_op = comp.create_component_from_func(export_model,
                                            packages_to_install=[],
                                            base_image='quay.io/ltomasbo/sparseml')
eval_op = comp.create_component_from_func(eval_model,
                                          packages_to_install=["datasets"],
                                          base_image='quay.io/ltomasbo/sparseml:eval2')
upload_op = comp.create_component_from_func(upload_pruned_model,
                                            packages_to_install=["boto3"],
                                            base_image='registry.access.redhat.com/ubi9/python-311')

# Define your pipeline function
@dsl.pipeline(
    name="LLM Pruning Pipeline",
    description="A Pipeline for pruning LLMs with SparseML"
)
def sparseml_pipeline(
    model_name:str="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    accuracy:int=90,
    sparse:bool=True,
    eval:bool=False,
    eval_task:str="hellaswag",
    eval_batch_size:str="64",
):
    print("Params", model_name, accuracy)
    vol = V1Volume(
        name='models-shared',
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
            claim_name='models-shared',)
        )

    gpu_toleration = V1Toleration(effect='NoSchedule',
                                  key='nvidia.com/gpu',
                                  operator='Equal',
                                  value='true')

    # Directories
    BASE_DIR = "/mnt/models/"
    MODEL_DIR = BASE_DIR + "llm"
    SPARSE_MODEL_DIR = BASE_DIR + "sparse-llm"
    EXPORTED_MODEL_DIR = BASE_DIR + "exported"

    # Download volumes
    download_llm = download_op(model_name, destination_path=MODEL_DIR)
    download_llm.add_pvolumes({"/mnt/models": vol})

    sparse_llm = None
    export_llm = None
    upload_pruned_llm = None

    # Condition for sparse operation
    with dsl.Condition(sparse == True):
        sparse_llm = sparse_op(model_path=MODEL_DIR,
                              compress_model_path=SPARSE_MODEL_DIR,
                              #ds="open-platypus",
                              ds="garage-bAInd/Open-Platypus",
                              precision="bfloat16")
        sparse_llm.add_pvolumes({"/mnt/models": vol})
        sparse_llm.add_node_selector_constraint(label_name='nvidia.com/gpu.present', value='true')
        sparse_llm.add_toleration(gpu_toleration)
        sparse_llm.add_resource_request('nvidia.com/gpu', "1")
        sparse_llm.add_resource_limit('nvidia.com/gpu', "1")
        sparse_llm.after(download_llm)

        # Condition for eval operation
        with dsl.Condition(eval == True):
            eval_llm = eval_op(model_path=SPARSE_MODEL_DIR, tasks=eval_task, batch_size=eval_batch_size)
            eval_llm.add_pvolumes({"/mnt/models": vol})
            eval_llm.add_node_selector_constraint(label_name='nvidia.com/gpu.present', value='true')
            eval_llm.add_toleration(gpu_toleration)
            eval_llm.add_resource_request('nvidia.com/gpu', "1")
            eval_llm.add_resource_limit('nvidia.com/gpu', "1")
            eval_llm.after(sparse_llm)

            eval_llm_base = eval_op(model_path=MODEL_DIR, tasks=eval_task, batch_size=eval_batch_size)
            eval_llm_base.add_pvolumes({"/mnt/models": vol})
            eval_llm_base.add_node_selector_constraint(label_name='nvidia.com/gpu.present', value='true')
            eval_llm_base.add_toleration(gpu_toleration)
            eval_llm_base.add_resource_request('nvidia.com/gpu', "1")
            eval_llm_base.add_resource_limit('nvidia.com/gpu', "1")
            eval_llm_base.after(download_llm)

            export_llm = export_op(model_path=SPARSE_MODEL_DIR, exported_model_path=EXPORTED_MODEL_DIR)
            export_llm.add_pvolumes({"/mnt/models": vol})
            export_llm.after(eval_llm)

            upload_pruned_llm = upload_op(model_path=EXPORTED_MODEL_DIR)
            upload_pruned_llm.add_env_variable(env_from_secret('s3_access_key', 'aws-connection-models', 'AWS_ACCESS_KEY_ID'))
            upload_pruned_llm.add_env_variable(env_from_secret('s3_secret_access_key', 'aws-connection-models', 'AWS_SECRET_ACCESS_KEY'))
            upload_pruned_llm.add_env_variable(env_from_secret('s3_host', 'aws-connection-models', 'AWS_S3_ENDPOINT'))
            upload_pruned_llm.add_env_variable(env_from_secret('s3_bucket', 'aws-connection-models', 'AWS_S3_BUCKET'))
            upload_pruned_llm.add_pvolumes({"/mnt/models": vol})
            upload_pruned_llm.after(export_llm)

        with dsl.Condition(eval == False):
            export_llm = export_op(model_path=SPARSE_MODEL_DIR, exported_model_path=EXPORTED_MODEL_DIR)
            export_llm.add_pvolumes({"/mnt/models": vol})
            export_llm.after(sparse_llm)

            upload_pruned_llm = upload_op(model_path=EXPORTED_MODEL_DIR)
            upload_pruned_llm.add_env_variable(env_from_secret('s3_access_key', 'aws-connection-models', 'AWS_ACCESS_KEY_ID'))
            upload_pruned_llm.add_env_variable(env_from_secret('s3_secret_access_key', 'aws-connection-models', 'AWS_SECRET_ACCESS_KEY'))
            upload_pruned_llm.add_env_variable(env_from_secret('s3_host', 'aws-connection-models', 'AWS_S3_ENDPOINT'))
            upload_pruned_llm.add_env_variable(env_from_secret('s3_bucket', 'aws-connection-models', 'AWS_S3_BUCKET'))
            upload_pruned_llm.add_pvolumes({"/mnt/models": vol})
            upload_pruned_llm.after(export_llm)

    with dsl.Condition(sparse == False):
        with dsl.Condition(eval == True):
            eval_llm_base = eval_op(model_path=MODEL_DIR, tasks=eval_task, batch_size=eval_batch_size)
            eval_llm_base.add_pvolumes({"/mnt/models": vol})
            eval_llm_base.add_node_selector_constraint(label_name='nvidia.com/gpu.present', value='true')
            eval_llm_base.add_toleration(gpu_toleration)
            eval_llm_base.add_resource_request('nvidia.com/gpu', "1")
            eval_llm_base.add_resource_limit('nvidia.com/gpu', "1")
            eval_llm_base.after(download_llm)

        export_llm = export_op(model_path=MODEL_DIR, exported_model_path=EXPORTED_MODEL_DIR)
        export_llm.add_pvolumes({"/mnt/models": vol})
        export_llm.after(download_llm)

        upload_pruned_llm = upload_op(model_path=EXPORTED_MODEL_DIR)
        upload_pruned_llm.add_env_variable(env_from_secret('s3_access_key', 'aws-connection-models', 'AWS_ACCESS_KEY_ID'))
        upload_pruned_llm.add_env_variable(env_from_secret('s3_secret_access_key', 'aws-connection-models', 'AWS_SECRET_ACCESS_KEY'))
        upload_pruned_llm.add_env_variable(env_from_secret('s3_host', 'aws-connection-models', 'AWS_S3_ENDPOINT'))
        upload_pruned_llm.add_env_variable(env_from_secret('s3_bucket', 'aws-connection-models', 'AWS_S3_BUCKET'))
        upload_pruned_llm.add_pvolumes({"/mnt/models": vol})
        upload_pruned_llm.after(export_llm)

# Compile the pipeline
TektonCompiler().compile(sparseml_pipeline, 'sparseml_pipeline_custom.yaml')
