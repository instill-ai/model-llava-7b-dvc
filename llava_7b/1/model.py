# pylint: skip-file
import random

import ray
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import numpy as np

from instill.helpers.const import DataType, VisualQuestionAnsweringInput
from instill.helpers.ray_io import StandardTaskIO
from instill.helpers.ray_config import (
    instill_deployment,
    get_compose_ray_address,
    InstillDeployable,
)

from ray_pb2 import (
    ModelReadyRequest,
    ModelReadyResponse,
    ModelMetadataRequest,
    ModelMetadataResponse,
    ModelInferRequest,
    ModelInferResponse,
    InferTensor,
)

ray.init(address=get_compose_ray_address(10001))
# this import must come after `ray.init()`
from ray import serve

import json


@instill_deployment
class Llava:
    def __init__(self, model_path: str):
        # self.application_name = "_".join(model_path.split("/")[3:5])
        # self.deployement_name = model_path.split("/")[4]
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # device_map="auto",
            # use_flash_attention_2=True # should instill with `flask-attn`
            # load_in_4bit=True # should instaill with `bitsandbytes`
        ).to(
            0
        )  # ???

        self.processor = AutoProcessor.from_pretrained(model_path)

    def ModelMetadata(self, req: ModelMetadataRequest) -> ModelMetadataResponse:
        resp = ModelMetadataResponse(
            name=req.name,
            versions=req.version,
            framework="python",
            inputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="prompt_image",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="max_new_tokens",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="temperature",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="top_k",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="random_seed",
                    datatype=str(DataType.TYPE_UINT64.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="extra_params",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1, -1],
                ),
            ],
        )
        return resp

    def ModelReady(self, req: ModelReadyRequest) -> ModelReadyResponse:
        resp = ModelReadyResponse(ready=True)
        return resp

    async def ModelInfer(self, request: ModelInferRequest) -> ModelInferResponse:
        resp = ModelInferResponse(
            model_name=request.model_name,
            model_version=request.model_version,
            outputs=[],
            raw_output_contents=[],
        )

        task_visual_question_answering_input: VisualQuestionAnsweringInput = (
            StandardTaskIO.parse_task_visual_question_answering_input(request=request)
        )

        if task_visual_question_answering_input.temperature <= 0.0:
            task_visual_question_answering_input.temperature = 0.8

        # Initialize random seed
        if task_visual_question_answering_input.random_seed > 0:
            random.seed(task_visual_question_answering_input.random_seed)
            np.random.seed(task_visual_question_answering_input.random_seed)
            torch.manual_seed(task_visual_question_answering_input.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    task_visual_question_answering_input.random_seed
                )

        processed_prompt = (
            f"USER: <image>\n{task_visual_question_answering_input.prompt}\nASSISTANT:"
        )
        raw_image = task_visual_question_answering_input.prompt_image
        inputs = self.processor(processed_prompt, raw_image, return_tensors="pt").to(
            0, torch.float16
        )

        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

        sequences = [
            self.processor.decode(output[0][2:], skip_special_tokens=True),
        ]

        task_visual_question_answering_output = (
            StandardTaskIO.parse_task_visual_question_answering_output(
                sequences=sequences
            )
        )

        resp.outputs.append(
            InferTensor(
                name="text",
                shape=[1, len(sequences)],
                datatype=str(DataType.TYPE_STRING),
            )
        )

        resp.raw_output_contents.append(task_visual_question_answering_output)

        return resp


deployable = InstillDeployable(Llava, model_weight_or_folder_name="llava-1.5-7b-hf")

# you can also have a fine-grained control of the cpu and gpu resources allocation
deployable.update_num_cpus(4)
deployable.update_num_gpus(1)
