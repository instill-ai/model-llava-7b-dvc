# pylint: skip-file
import os
import io
import base64
from json.decoder import JSONDecodeError
from typing import List

from PIL import Image

import random

# TORCH_GPU_MEMORY_FRACTION = 0.95  # Target memory ~= 15G on 16G card
TORCH_GPU_MEMORY_FRACTION = 0.38  # Target memory ~= 15G on 40G card
TORCH_GPU_DEVICE_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{TORCH_GPU_DEVICE_ID}"


import traceback

import io
import time
import json
import base64
from pathlib import Path
from PIL import Image

import traceback

import numpy as np
from typing import Any, Dict, List, Union

import transformers
from transformers import AutoTokenizer
import torch

torch.cuda.set_per_process_memory_fraction(
    TORCH_GPU_MEMORY_FRACTION, 0  # it count of number of device instead of device index
)

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

from llava.utils import disable_torch_init
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, Conversation, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token


class VisualQuestionAnsweringInput:
    prompt = ""
    prompt_images: Union[List[np.ndarray], None] = None
    chat_history: Union[List[str], None] = None
    system_message: Union[str, None] = None
    max_new_tokens = 100
    temperature = 0.8
    top_k = 1
    random_seed = 0
    stop_words: Any = ""  # Optional
    extra_params: Dict[str, str] = {}


class TritonPythonModel:
    # Reference: https://docs.nvidia.com/launchpad/data-science/sentiment/latest/sentiment-triton-overview.html
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
        Both keys and values are strings. The dictionary keys and values are:
        * model_config: A JSON string containing the model configuration
        * model_instance_kind: A string containing model instance kind
        * model_instance_device_id: A string containing model instance device ID
        * model_repository: Model repository path
        * model_version: Model version
        * model_name: Model name
        """
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])

        # Load the model
        model_path = str(Path(__file__).parent.absolute().joinpath("llava-v1.5-7b"))
        self.logger.log_info(f"[DEBUG] load model under path: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.logger.log_info(
            f"[DEBUG] self.tokenizer.pad_token: {self.tokenizer.pad_token}"
        )
        self.logger.log_info(
            f"[DEBUG] self.tokenizer.eos_token: {self.tokenizer.eos_token}"
        )
        self.logger.log_info(
            f"[DEBUG] transformers version: {transformers.__version__}"
        )
        self.logger.log_info(f"[DEBUG] torch version: {torch.__version__}")

        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="auto",  # "cpu"
            # max_memory={0: "12GB", 1: "12GB", 2: "12GB", 3: "12GB"},
            torch_dtype=torch.float16,
        )

        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "text")
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        # disable_torch_init()
        responses = []
        for request in requests:
            visual_question_answering_input = VisualQuestionAnsweringInput()
            if pb_utils.get_input_tensor_by_name(request, "prompt") is not None:
                visual_question_answering_input.prompt = str(
                    pb_utils.get_input_tensor_by_name(request, "prompt")
                    .as_numpy()[0]
                    .decode("utf-8")
                )
            else:
                raise ValueError("Prompt must be non-empty")

            if pb_utils.get_input_tensor_by_name(request, "prompt_images") is not None:
                input_tensors = pb_utils.get_input_tensor_by_name(
                    request, "prompt_images"
                ).as_numpy()
                images = []
                for enc in input_tensors:
                    if len(enc) == 0:
                        continue
                    try:
                        enc_json = json.loads(str(enc.decode("utf-8")))
                        if len(enc_json) == 0:
                            continue
                        decoded_enc = enc_json[0]
                    except JSONDecodeError:
                        print("[DEBUG] WARNING `enc_json` parsing faield!")
                    # pil_img = Image.open(io.BytesIO(enc.astype(bytes)))
                    pil_img = Image.open(io.BytesIO(base64.b64decode(decoded_enc)))
                    image = np.array(pil_img)
                    if len(image.shape) == 2:  # gray image
                        raise ValueError(
                            f"The image shape with {image.shape} is "
                            f"not in acceptable"
                        )
                    images.append(image)
                visual_question_answering_input.prompt_images = images

            # TODO: Support chat_history in next version
            # if pb_utils.get_input_tensor_by_name(request, "chat_history") is not None:
            #     chat_history_str = str(
            #         pb_utils.get_input_tensor_by_name(request, "chat_history")
            #         .as_numpy()[0]
            #         .decode("utf-8")
            #     )
            #     try:
            #         visual_question_answering_input.chat_history = json.loads(chat_history_str)
            #     except json.decoder.JSONDecodeError:
            #         pass

            if pb_utils.get_input_tensor_by_name(request, "system_message") is not None:
                visual_question_answering_input.system_message = str(
                    pb_utils.get_input_tensor_by_name(request, "system_message")
                    .as_numpy()[0]
                    .decode("utf-8")
                )
                if len(visual_question_answering_input.system_message) == 0:
                    visual_question_answering_input.system_message = None

            if pb_utils.get_input_tensor_by_name(request, "max_new_tokens") is not None:
                visual_question_answering_input.max_new_tokens = int(
                    pb_utils.get_input_tensor_by_name(
                        request, "max_new_tokens"
                    ).as_numpy()[0]
                )

            if pb_utils.get_input_tensor_by_name(request, "top_k") is not None:
                visual_question_answering_input.top_k = int(
                    pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()[0]
                )

            if pb_utils.get_input_tensor_by_name(request, "temperature") is not None:
                visual_question_answering_input.temperature = round(
                    float(
                        pb_utils.get_input_tensor_by_name(
                            request, "temperature"
                        ).as_numpy()[0]
                    ),
                    2,
                )

            if pb_utils.get_input_tensor_by_name(request, "random_seed") is not None:
                visual_question_answering_input.random_seed = int(
                    pb_utils.get_input_tensor_by_name(
                        request, "random_seed"
                    ).as_numpy()[0]
                )

            if pb_utils.get_input_tensor_by_name(request, "extra_params") is not None:
                extra_params_str = str(
                    pb_utils.get_input_tensor_by_name(request, "extra_params")
                    .as_numpy()[0]
                    .decode("utf-8")
                )
                try:
                    visual_question_answering_input.extra_params = json.loads(
                        extra_params_str
                    )
                except json.decoder.JSONDecodeError:
                    pass

            print(
                f"Before Preprocessing `prompt`        : {type(visual_question_answering_input.prompt)}. {visual_question_answering_input.prompt}"
            )
            print(
                f"Before Preprocessing `prompt_images` : {type(visual_question_answering_input.prompt_images)}. {visual_question_answering_input.prompt_images}"
            )
            print(
                f"Before Preprocessing `chat_history`  : {type(visual_question_answering_input.chat_history)}. {visual_question_answering_input.chat_history}"
            )
            print(
                f"Before Preprocessing `system_message`: {type(visual_question_answering_input.system_message)}. {visual_question_answering_input.system_message}"
            )
            print(
                f"Before Preprocessing `max_new_tokens`: {type(visual_question_answering_input.max_new_tokens)}. {visual_question_answering_input.max_new_tokens}"
            )
            print(
                f"Before Preprocessing `temperature`   : {type(visual_question_answering_input.temperature)}. {visual_question_answering_input.temperature}"
            )
            print(
                f"Before Preprocessing `top_k`         : {type(visual_question_answering_input.top_k)}. {visual_question_answering_input.top_k}"
            )
            print(
                f"Before Preprocessing `random_seed`   : {type(visual_question_answering_input.random_seed)}. {visual_question_answering_input.random_seed}"
            )
            print(
                f"Before Preprocessing `stop_words`    : {type(visual_question_answering_input.stop_words)}. {visual_question_answering_input.stop_words}"
            )
            print(
                f"Before Preprocessing `extra_params`  : {type(visual_question_answering_input.extra_params)}. {visual_question_answering_input.extra_params}"
            )

            # Preprocessing
            if visual_question_answering_input.random_seed > 0:
                random.seed(visual_question_answering_input.random_seed)
                np.random.seed(visual_question_answering_input.random_seed)
                torch.manual_seed(visual_question_answering_input.random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(
                        visual_question_answering_input.random_seed
                    )

            # Handle Conversation
            conv_mode = "llava_v1"
            prompt_roles = ["USER", "ASSISTANT", "SYSTEM"]
            if (
                visual_question_answering_input.chat_history is not None
                and len(visual_question_answering_input.chat_history) > 0
            ):
                prompt_conversation = []
                default_system_message = visual_question_answering_input.system_message
                for chat_entity in visual_question_answering_input.chat_history:
                    role = str(chat_entity["role"]).upper()
                    chat_history_messages = None
                    chat_hisotry_images = []

                    for chat_entity_message in chat_entity["content"]:
                        if chat_entity_message["type"] == "text":
                            if chat_history_messages is not None:
                                raise ValueError(
                                    "Multiple text message detected"
                                    " in a single chat history entity"
                                )
                            # [{'role': 'system', 'content': [{'type': 'text', 'Content': {'Text': "What's in this image?"}}]}]
                            chat_history_messages = chat_entity_message["Content"][
                                "Text"
                            ]
                        elif chat_entity_message["type"] == "image_url":
                            # TODO: imeplement image parser in model_backedn
                            # This field is expected to be base64 encoded string
                            IMAGE_BASE64_PREFIX = (
                                "data:image/jpeg;base64,"  # "{base64_image}"
                            )

                            if len(chat_entity_message["Content"]["ImageUrl"]) == 0:
                                continue
                            elif (
                                "promptImageUrl"
                                in chat_entity_message["Content"]["ImageUrl"][
                                    "image_url"
                                ]["Type"]
                            ):
                                image = Image.open(
                                    io.BytesIO(
                                        requests.get(
                                            chat_entity_message["Content"]["ImageUrl"][
                                                "image_url"
                                            ]["Type"]["promptImageUrl"]
                                        ).content
                                    )
                                )
                                chat_hisotry_images.append(image)
                            elif (
                                "promptImageBase64"
                                in chat_entity_message["Content"]["ImageUrl"][
                                    "image_url"
                                ]["Type"]
                            ):
                                image_base64_str = chat_entity_message["Content"][
                                    "ImageUrl"
                                ]["image_url"]["Type"]["promptImageBase64"]
                                if image_base64_str.startswith(IMAGE_BASE64_PREFIX):
                                    image_base64_str = image_base64_str[
                                        IMAGE_BASE64_PREFIX:
                                    ]
                                # expected content in url with base64 format:
                                # f"data:image/jpeg;base64,{base64_image}"
                                pil_img = Image.open(
                                    io.BytesIO(base64.b64decode(image_base64_str))
                                )
                                image = np.array(pil_img)
                                if len(image.shape) == 2:  # gray image
                                    raise ValueError(
                                        f"The chat history image shape with {image.shape} is "
                                        f"not in acceptable"
                                    )
                                chat_hisotry_images.append(image)
                        else:
                            raise ValueError(
                                "Unsupported chat_hisotry message type"
                                ", expected eithjer 'text' or 'image_url'"
                                f" but get {chat_entity_message['type']}"
                            )

                    # TODO: support image message in chat history
                    # self.messages.append([role, message])
                    if role not in prompt_roles:
                        raise ValueError(
                            f"Role `{chat_entity['role']}` is not in supported"
                            f"role list ({','.join(prompt_roles)})"
                        )
                    elif (
                        role == prompt_roles[-1] and default_system_message is not None
                    ):
                        raise ValueError(
                            "it's ambiguious to set `system_message` and "
                            f"using role `{prompt_roles[-1]}` simultaneously"
                        )
                    elif chat_history_messages is None:
                        raise ValueError(
                            f"No message found in chat_history. {chat_entity_message}"
                        )
                    if role == prompt_roles[-1]:
                        default_system_message = chat_history_messages
                    else:
                        prompt_conversation.append([role, chat_history_messages])

                if default_system_message is None:
                    default_system_message = (
                        "A chat between a curious human and an artificial intelligence assistant. "
                        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
                    )
                conv = Conversation(
                    system=default_system_message,
                    roles=tuple(prompt_roles[:-1]),
                    version="v1",
                    messages=prompt_conversation,
                    offset=0,
                    sep_style=SeparatorStyle.LLAMA_2,
                    sep="<s>",
                    sep2="</s>",
                )
                # conv.append_message(conv.roles[0], visual_question_answering_input.prompt)
            else:
                if visual_question_answering_input.system_message is not None:
                    conv = Conversation(
                        system=visual_question_answering_input.system_message,
                        roles=tuple(prompt_roles[:-1]),
                        version="v1",
                        messages=[],
                        offset=0,
                        sep_style=SeparatorStyle.LLAMA_2,
                        sep="<s>",
                        sep2="</s>",
                    )
                else:
                    conv = conv_templates[conv_mode].copy()
                # conv.append_message(conv.roles[0], visual_question_answering_input.prompt)

            # Handle Image
            vision_tower = self.model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower = vision_tower.to(device="cuda", dtype=torch.float16)
            image_processor = vision_tower.image_processor

            if len(visual_question_answering_input.prompt_images) > 0:
                # Currently only support 1 image
                image_tensor = process_images(
                    [visual_question_answering_input.prompt_images[0]],
                    image_processor,
                    {"image_aspect_ratio": "pad"},
                ).to(self.model.device, dtype=torch.float16)
            else:
                self.logger.log_info(f"[WARNGING], NO IMAGE received")

            # if image is not None:
            # TODO: Support mulitple image token
            inp = DEFAULT_IMAGE_TOKEN + "\n" + visual_question_answering_input.prompt
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)

            target_prompt = conv.get_prompt()
            print("----------------")
            print(f"[DEBUG] Conversation Prompt: \n{target_prompt}")
            print("----------------")

            input_ids = (
                tokenizer_image_token(
                    target_prompt,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .cuda()
            )

            # Inference
            # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
            # https://huggingface.co/docs/transformers/v4.30.1/en/main_classes/text_generation#transformers.GenerationConfig

            t0 = time.time()  # calculate time cost in following function call

            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=visual_question_answering_input.temperature,
                top_k=visual_question_answering_input.top_k,
                max_new_tokens=visual_question_answering_input.max_new_tokens,
                use_cache=False,
                **visual_question_answering_input.extra_params,
            )
            self.logger.log_info(
                f"Inference time cost {time.time()-t0}s with input lenth {len(visual_question_answering_input.prompt)}"
            )

            outputs = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
            ).strip()

            text_outputs = [
                outputs,
            ]
            triton_output_tensor = pb_utils.Tensor(
                "text", np.asarray(text_outputs, dtype=self.output0_dtype)
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[triton_output_tensor])
            )
        return responses

    def finalize(self):
        self.logger.log_info("Issuing finalize to Llava Model Transformer backend")
