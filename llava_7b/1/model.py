# pylint: skip-file
import random

import ray
import torch
import transformers
from transformers import AutoTokenizer
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


import traceback

import io
import time
import json
import base64
from pathlib import Path
from PIL import Image


from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, Conversation, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token


@instill_deployment
class Llava:
    def __init__(self, model_path: str):
        # self.application_name = "_".join(model_path.split("/")[3:5])
        # self.deployement_name = model_path.split("/")[4]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
            # torch_dtype=torch.float16,
            # low_cpu_mem_usage=True
        )

        # TODO: move to transformer pipeline
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

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

        # Preparing for the prompt
        conv_mode = "llava_v1"
        prompt_in_conversation = False
        try:
            parsed_conversation = json.loads(
                task_visual_question_answering_input.prompt
            )
            # using fixed roles
            roles = ["USER", "ASSISTANT"]
            roles_lookup = {x: i for i, x in enumerate(roles)}
            conv = None
            for i, x in enumerate(parsed_conversation):
                role = str(x["role"]).upper()
                self.logger.log_info(f'[DEBUG] Message {i}: {role}: {x["content"]}')
                if i == 0:
                    if role == "SYSTEM":
                        conv = Conversation(
                            system=str(x["content"]),
                            roles=("USER", "ASSISTANT"),
                            version="llama_v2",
                            messages=[],
                            offset=0,
                            sep_style=SeparatorStyle.LLAMA_2,
                            sep="<s>",
                            sep2="</s>",
                        )
                    else:
                        conv = conv_templates[conv_mode].copy()
                        conv.roles = tuple(roles)
                        conv.append_message(
                            conv.roles[roles_lookup[role]], x["content"]
                        )
                else:
                    conv.append_message(conv.roles[roles_lookup[role]], x["content"])
            prompt_in_conversation = True
        except json.decoder.JSONDecodeError:
            pass

        if not prompt_in_conversation or conv is None:
            conv = conv_templates[conv_mode].copy()

        # Preparing for the image
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower = vision_tower.to(device="cuda", dtype=torch.float16)
        image_processor = vision_tower.image_processor

        image_tensor = process_images(
            [task_visual_question_answering_input.image],
            image_processor,
            {"image_aspect_ratio": "pad"},
        ).to(self.model.device, dtype=torch.float16)

        # add image to prompt
        inp = DEFAULT_IMAGE_TOKEN + "\n" + task_visual_question_answering_input.prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        target_prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                target_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        if task_visual_question_answering_input.temperature <= 0:
            task_visual_question_answering_input = 0.8

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=task_visual_question_answering_input.temperature,
            top_k=task_visual_question_answering_input.top_k,
            max_new_tokens=task_visual_question_answering_input.max_new_tokens,
            use_cache=False
            # **extra_params
        )

        # output_ids[output_ids == -200] = 0
        outputs = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        sequences = [
            outputs,
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


deployable = InstillDeployable(Llava, model_weight_or_folder_name="llava-v1.5-7b")

# you can also have a fine-grained control of the cpu and gpu resources allocation
deployable.update_num_cpus(4)
deployable.update_num_gpus(1)
