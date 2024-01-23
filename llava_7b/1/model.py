# pylint: skip-file
import os

TORCH_GPU_DEVICE_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{TORCH_GPU_DEVICE_ID}"


import io
import time
import requests
import random
import base64
import ray
import torch
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

import numpy as np

from instill.helpers.const import DataType, VisualQuestionAnsweringInput
from instill.helpers.ray_io import (
    serialize_byte_tensor,
    deserialize_bytes_tensor,
    StandardTaskIO,
)

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    construct_infer_response,
    construct_metadata_response,
    Metadata,
)


from conversation import Conversation, conv_templates, SeparatorStyle


@instill_deployment
class Llava:
    def __init__(self, model_path: str):
        self.application_name = "_".join(model_path.split("/")[3:5])
        self.deployement_name = model_path.split("/")[4]
        print(f"application_name: {self.application_name}")
        print(f"deployement_name: {self.deployement_name}")
        print(f"torch version: {torch.__version__}")

        print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count() : {torch.cuda.device_count()}")
        print(f"torch.cuda.current_device() : {torch.cuda.current_device()}")
        print(f"torch.cuda.device(0) : {torch.cuda.device(0)}")
        print(f"torch.cuda.get_device_name(0) : {torch.cuda.get_device_name(0)}")

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # device_map="auto",
            # use_flash_attention_2=True # should instill with `flask-attn`
            # load_in_4bit=True # should instaill with `bitsandbytes`
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_path)

    def ModelMetadata(self, req):
        resp = construct_metadata_response(
            req=req,
            inputs=[
                Metadata(
                    name="prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="prompt_images",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="chat_history",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="system_message",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="max_new_tokens",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="temperature",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[1],
                ),
                Metadata(
                    name="top_k",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="random_seed",
                    datatype=str(DataType.TYPE_UINT64.name),
                    shape=[1],
                ),
                Metadata(
                    name="extra_params",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                Metadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1, -1],
                ),
            ],
        )
        return resp

    async def __call__(self, req):
        task_visual_question_answering_input: VisualQuestionAnsweringInput = (
            StandardTaskIO.parse_task_visual_question_answering_input(request=req)
        )

        print("----------------________")
        print(task_visual_question_answering_input)
        print("----------------________")

        print("print(task_text_generation_chat.prompt")
        print(task_visual_question_answering_input.prompt)
        print("-------\n")

        print("print(task_text_generation_chat.prompt_images")
        print(task_visual_question_answering_input.prompt_images)
        print("-------\n")

        print("print(task_text_generation_chat.chat_history")
        print(task_visual_question_answering_input.chat_history)
        print("-------\n")

        print("print(task_text_generation_chat.system_message")
        print(task_visual_question_answering_input.system_message)
        if len(task_visual_question_answering_input.system_message) is not None:
            if len(task_visual_question_answering_input.system_message) == 0:
                task_visual_question_answering_input.system_message = None
        print("-------\n")

        print("print(task_text_generation_chat.max_new_tokens")
        print(task_visual_question_answering_input.max_new_tokens)
        print("-------\n")

        print("print(task_text_generation_chat.temperature")
        print(task_visual_question_answering_input.temperature)
        print("-------\n")

        print("print(task_text_generation_chat.top_k")
        print(task_visual_question_answering_input.top_k)
        print("-------\n")

        print("print(task_text_generation_chat.random_seed")
        print(task_visual_question_answering_input.random_seed)
        print("-------\n")

        print("print(task_text_generation_chat.stop_words")
        print(task_visual_question_answering_input.stop_words)
        print("-------\n")

        print("print(task_text_generation_chat.extra_params")
        print(task_visual_question_answering_input.extra_params)
        print("-------\n")

        if task_visual_question_answering_input.temperature <= 0.0:
            task_visual_question_answering_input.temperature = 0.8

        if task_visual_question_answering_input.random_seed > 0:
            random.seed(task_visual_question_answering_input.random_seed)
            np.random.seed(task_visual_question_answering_input.random_seed)

        # Process chat_history
        # Preprocessing
        CHECK_FIRST_ROLE_IS_USER = False
        COMBINED_CONSEQUENCE_PROMPTS = True
        conv_mode = "llava_v1"
        prompt_roles = ["USER", "ASSISTANT", "SYSTEM"]
        conversation_prompt = task_visual_question_answering_input.prompt
        if (
            task_visual_question_answering_input.chat_history is not None
            and len(task_visual_question_answering_input.chat_history) > 0
        ):
            prompt_conversation = []
            default_system_message = task_visual_question_answering_input.system_message
            for chat_entity in task_visual_question_answering_input.chat_history:
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
                        # This structure comes from google protobuf `One of` Syntax, where an additional layer in Content
                        # [{'role': 'system', 'content': [{'type': 'text', 'Content': {'Text': "What's in this image?"}}]}]
                        if "Content" in chat_entity_message:
                            chat_history_messages = chat_entity_message["Content"][
                                "Text"
                            ]
                        elif "Text" in chat_entity_message:
                            chat_history_messages = chat_entity_message["Text"]
                        elif "text" in chat_entity_message:
                            chat_history_messages = chat_entity_message["text"]
                        else:
                            raise ValueError(
                                f"Unknown structure of chat_hisoty: {task_visual_question_answering_input.chat_history}"
                            )
                    elif chat_entity_message["type"] == "image_url":
                        # TODO: imeplement image parser in model_backedn
                        # This field is expected to be base64 encoded string
                        IMAGE_BASE64_PREFIX = (
                            "data:image/jpeg;base64,"  # "{base64_image}"
                        )
                        # This structure comes from google protobuf `One of` Syntax, where an additional layer in Content
                        # TODO: Handling this field
                        if (
                            "Content" not in chat_entity_message
                            or "ImageUrl" not in chat_entity_message["Content"]
                        ):
                            print(
                                f"Unsupport chat_entity_message format: {chat_entity_message}"
                            )
                            continue

                        if len(chat_entity_message["Content"]["ImageUrl"]) == 0:
                            continue
                        elif (
                            "promptImageUrl"
                            in chat_entity_message["Content"]["ImageUrl"]["image_url"][
                                "Type"
                            ]
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
                            in chat_entity_message["Content"]["ImageUrl"]["image_url"][
                                "Type"
                            ]
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
                    role == prompt_roles[-1]
                    and default_system_message is not None
                    and len(default_system_message) > 0
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
                    if CHECK_FIRST_ROLE_IS_USER:
                        if len(prompt_conversation) == 0 and role != prompt_roles[0]:
                            prompt_conversation.append([prompt_roles[0], " "])
                    if COMBINED_CONSEQUENCE_PROMPTS:
                        if (
                            len(prompt_conversation) > 0
                            and prompt_conversation[-1][0] == role
                        ):
                            laset_conversation = prompt_conversation.pop()
                            chat_history_messages = (
                                f"{laset_conversation[1]}\n\n{chat_history_messages}"
                            )
                    prompt_conversation.append([role, chat_history_messages])

            if default_system_message is None:
                default_system_message = (
                    "A chat between a curious human and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
                )

            if COMBINED_CONSEQUENCE_PROMPTS:
                if (
                    len(prompt_conversation) > 0
                    and prompt_conversation[-1][0] == prompt_roles[0]
                ):
                    laset_conversation = prompt_conversation.pop()
                    conversation_prompt = (
                        f"{laset_conversation[1]}\n\n{conversation_prompt}"
                    )

            conv = Conversation(
                system=default_system_message,
                roles=tuple(prompt_roles[:-1]),
                version="v1",
                messages=prompt_conversation,
                offset=0,
                sep_style=SeparatorStyle.TWO,
                sep=" ",
                sep2="</s>",
            )
            # for llava model, handle first prompt later
            # conv.append_message(conv.roles[0], conversation_prompt)
        else:
            if task_visual_question_answering_input.system_message is not None:
                conv = Conversation(
                    system=task_visual_question_answering_input.system_message,
                    roles=tuple(prompt_roles[:-1]),
                    version="v1",
                    messages=[],
                    offset=0,
                    sep_style=SeparatorStyle.TWO,
                    sep=" ",
                    sep2="</s>",
                )
            else:
                conv = conv_templates[conv_mode].copy()
            # for llava model, handle first prompt later
            # conv.append_message(conv.roles[0], task_visual_question_answering_input.prompt)

        # Llava Prompt be like:
        # f"USER: <image>\n{task_visual_question_answering_input.prompt}\nASSISTANT:"
        if len(task_visual_question_answering_input.prompt_images) > 0:
            conv.append_message(conv.roles[0], f"<image>\n{conversation_prompt}")
        else:
            self.logger.log_info(f"[WARNGING], NO IMAGE received")
            conv.append_message(conv.roles[0], conversation_prompt)
        conv.append_message(conv.roles[1], None)

        processed_prompt = conv.get_prompt()
        input_length = len(processed_prompt) - len("<image>") + 1
        print(f"----------------, length: {input_length}")
        print(f"[DEBUG] Conversation Prompt: \n{conv.get_prompt()}")
        print("----------------")

        # Handle Images
        raw_image = None
        if len(task_visual_question_answering_input.prompt_images) > 0:
            raw_image = task_visual_question_answering_input.prompt_images[0]

        inputs = self.processor(processed_prompt, raw_image, return_tensors="pt").to(
            0, torch.float16
        )

        # End of Process chat_history
        t0 = time.time()
        output = self.model.generate(
            **inputs,
            max_new_tokens=task_visual_question_answering_input.max_new_tokens,
            do_sample=True,
            temperature=task_visual_question_answering_input.temperature,
            top_k=task_visual_question_answering_input.top_k,
            **task_visual_question_answering_input.extra_params,
        )
        print(f"Inference time cost {time.time()-t0}s")

        max_output_len = 0
        text_outputs = []
        # Not iterate outputs
        # for seq in sequences:
        print("Output No Clean ----")
        print(self.processor.decode(output[0], skip_special_tokens=True))
        print("Output Clean ----")
        print(self.processor.decode(output[0], skip_special_tokens=True)[input_length:])
        print("---")
        generated_text = (
            self.processor.decode(output[0], skip_special_tokens=True)[input_length:]
            .strip()
            .encode("utf-8")
        )
        text_outputs.append(generated_text)
        max_output_len = max(max_output_len, len(generated_text))
        # --
        text_outputs_len = len(text_outputs)
        task_output = serialize_byte_tensor(np.asarray(text_outputs))
        # task_output = StandardTaskIO.parse_task_text_generation_output(sequences)

        print("Output:")
        print(task_output)
        print(type(task_output))

        return construct_infer_response(
            req=req,
            outputs=[
                Metadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[text_outputs_len, max_output_len],
                )
            ],
            raw_outputs=[task_output],
        )


deployable = InstillDeployable(
    Llava, model_weight_or_folder_name="llava-1.5-7b-hf/", use_gpu=True
)

# # Optional
# deployable.update_max_replicas(2)
# deployable.update_min_replicas(0)
