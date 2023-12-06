# pylint: skip-file
import os
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

import numpy as np
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
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token
)

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
        # model_path = str(Path(__file__).parent.absolute().joinpath('llava-v1.5-13b'))
        model_path = str(Path(__file__).parent.absolute().joinpath('llava-v1.5-7b'))
        self.logger.log_info(f'[DEBUG] load model under path: {model_path}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        )
        self.logger.log_info(f'[DEBUG] self.tokenizer.pad_token: {self.tokenizer.pad_token}')
        self.logger.log_info(f'[DEBUG] self.tokenizer.eos_token: {self.tokenizer.eos_token}')
        self.logger.log_info(f'[DEBUG] transformers version: {transformers.__version__}')
        self.logger.log_info(f'[DEBUG] torch version: {torch.__version__}')
        
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="auto", # "cpu"
            # max_memory={0: "12GB", 1: "12GB", 2: "12GB", 3: "12GB"},
            torch_dtype=torch.float16
        )

        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "text")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        # disable_torch_init()
        responses = []

        for request in requests:
            try:
                prompt = str(pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0].decode("utf-8"))
                self.logger.log_info(f'[DEBUG] input `prompt` type({type(prompt)}): {prompt}')

                prompt_image = pb_utils.get_input_tensor_by_name(request, "prompt_image").as_numpy()[0]
                self.logger.log_info(f'[DEBUG] input `prompt_image` type({type(prompt_image)}): {len(prompt_image)}')

                extra_params_str = ""
                if pb_utils.get_input_tensor_by_name(request, "extra_params") is not None:
                    extra_params_str = str(pb_utils.get_input_tensor_by_name(request, "extra_params").as_numpy()[0].decode("utf-8"))
                self.logger.log_info(f'[DEBUG] input `extra_params` type({type(extra_params_str)}): {extra_params_str}')

                extra_params = {}
                # TODO: Add a function handle penalty
                try:
                    extra_params = json.loads(extra_params_str)
                    if 'repetition_penalty' in extra_params:
                        self.logger.log_info('[DEBUG] WARNING `repetition_penalty` would crash transformerparsing faield!')
                        del extra_params['repetition_penalty']
                except json.decoder.JSONDecodeError:
                    self.logger.log_info('[DEBUG] WARNING `extra_params` parsing faield!')
                    pass

                max_new_tokens = 100
                if pb_utils.get_input_tensor_by_name(request, "max_new_tokens") is not None:
                    max_new_tokens = int(pb_utils.get_input_tensor_by_name(request, "max_new_tokens").as_numpy()[0])
                self.logger.log_info(f'[DEBUG] input `max_new_tokens` type({type(max_new_tokens)}): {max_new_tokens}')

                top_k = 30
                if pb_utils.get_input_tensor_by_name(request, "top_k") is not None:
                    top_k = int(pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()[0])
                self.logger.log_info(f'[DEBUG] input `top_k` type({type(top_k)}): {top_k}')

                temperature = 0.8
                if pb_utils.get_input_tensor_by_name(request, "temperature") is not None:
                    temperature = float(pb_utils.get_input_tensor_by_name(request, "temperature").as_numpy()[0])
                temperature = round(temperature, 2)
                self.logger.log_info(f'[DEBUG] input `temperature` type({type(temperature)}): {temperature}')

                random_seed = 0
                if pb_utils.get_input_tensor_by_name(request, "random_seed") is not None:
                    random_seed = int(pb_utils.get_input_tensor_by_name(request, "random_seed").as_numpy()[0])
                self.logger.log_info(f'[DEBUG] input `random_seed` type({type(random_seed)}): {random_seed}')

                if random_seed > 0:
                   random.seed(random_seed)
                   np.random.seed(random_seed)
                   torch.manual_seed(random_seed)
                   if torch.cuda.is_available():
                       torch.cuda.manual_seed_all(random_seed)

                # Handle Conversation
                # conv_mode = "llava_llama_2"
                conv_mode = "llava_v1"
                prompt_in_conversation = False
                try:
                    parsed_conversation = json.loads(prompt)
                    # using fixed roles
                    roles = ['USER', 'ASSISTANT']
                    roles_lookup = {x: i for i, x in enumerate(roles)}
                    conv = None
                    for i, x in enumerate(parsed_conversation):
                        role = str(x['role']).upper()
                        self.logger.log_info(f'[DEBUG] Message {i}: {role}: {x["content"]}')
                        if i == 0:
                            if role == 'SYSTEM':
                                conv = Conversation(
                                    system=str(x['content']),
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
                                conv.append_message(conv.roles[roles_lookup[role]], x['content'])
                        else:
                            conv.append_message(conv.roles[roles_lookup[role]], x['content'])
                    prompt_in_conversation = True
                except json.decoder.JSONDecodeError:
                    pass

                if not prompt_in_conversation or conv is None:
                    conv = conv_templates[conv_mode].copy()
                    
                # Handle Image
                # TODO: Option for only-text input
                # TODO: Check wether protobuf satisfy the format
                # TODO: Should we resize the image?
                # Handle image
                vision_tower = self.model.get_vision_tower()
                if not vision_tower.is_loaded:
                    vision_tower.load_model()
                vision_tower = vision_tower.to(device='cuda', dtype=torch.float16)
                image_processor = vision_tower.image_processor

                success_parsed_image = False
                try:
                    original_image = Image.open(
                        io.BytesIO(prompt_image)
                    )  # for instill vd3p
                    success_parsed_image = True
                except Exception as e:
                    print(e)
                    pass

                if not success_parsed_image:
                    try:
                        original_image = Image.open(
                            io.BytesIO(base64.b64decode(prompt_image))
                        )  # for test script
                        success_parsed_image = True
                    except Exception as e:
                        print(e)
                        pass

                if not success_parsed_image:
                    raise ValueError("Unable to parsed image")
                else:
                    image = original_image

                self.logger.log_info(f"np.array(image).shape: {np.array(image).shape}")
                # self.logger.log_info(f"self.model.device: {self.model.device}")
                image_tensor = process_images(
                    [image],
                    image_processor,
                    {"image_aspect_ratio": 'pad'}
                ).to(self.model.device, dtype=torch.float16)
                self.logger.log_info(f"image_tensor.shape: {image_tensor.shape}")


                # if image is not None:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)

                target_prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(
                    target_prompt,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors='pt'
                ).unsqueeze(0).cuda()

                # Inference
                # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
                # https://huggingface.co/docs/transformers/v4.30.1/en/main_classes/text_generation#transformers.GenerationConfig
                

                t0 = time.time() # calculate time cost in following function call

                for _ in range(50):
                    output_ids = self.model.generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        max_new_tokens=max_new_tokens,
                        use_cache=False,
                        **extra_params
                    )
                    self.logger.log_info(f'Inference time cost {time.time()-t0}s with input lenth {len(prompt)}')

                    # output_ids[output_ids == -200] = 0
                    outputs = self.tokenizer.decode(
                        output_ids[0, input_ids.shape[1]:],
                        skip_special_tokens = True
                    ).strip()

                    if len(outputs) > 1:
                        break

                text_outputs = [outputs, ]
                triton_output_tensor = pb_utils.Tensor(
                    "text", np.asarray(text_outputs, dtype=self.output0_dtype)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[triton_output_tensor]))

                # torch.cuda.empty_cache()
            except Exception as e:
                self.logger.log_info(f"Error generating stream: {e}")
                self.logger.log_info(f"{traceback.format_exc()}")

                error = pb_utils.TritonError(f"Error generating stream: {e}")
                triton_output_tensor = pb_utils.Tensor(
                    "text", np.asarray(["N/A"], dtype=self.output0_dtype)
                )
                response = pb_utils.InferenceResponse(
                    output_tensors=[triton_output_tensor], error=error
                )
                responses.append(response)
                self.logger.log_info("The model did not receive the expected inputs")
                raise e
            return responses

    def finalize(self):
        self.logger.log_info("Cleaning up ...")
