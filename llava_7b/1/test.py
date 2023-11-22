import torch
from pathlib import Path
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM 
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from PIL import Image

if __name__ == "__main__":
    disable_torch_init()
    model_path = str(Path(__file__).parent.absolute().joinpath('llava-v1.5-13b'))

    import time
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False
    )
    
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True, 
        device_map="auto", # "cpu", "cuda:0,1,2,3"
        #  max_memory={0: "10GB", 1: "10GB", 2: "48GB", 3: "48GB"}
        torch_dtype=torch.float16
    )
    print('[DEBUG] loading time: ', time.time() - t0)

    
    
    # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

    # mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    # tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    # model.resize_token_embeddings(len(tokenizer))
    image_processor = None

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor

    # Handle Conversation
    conv_mode = "llava_llama_2"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # Handle Image
    image = Image.open('/home/tonywang/Projects/tmp/new_model/mnt_models/llava_13b/1/sample1.png')
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    image_tensor = process_images([image], image_processor, {"image_aspect_ratio": 'pad'})
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    print('[DEBUG] type of image tensor:', type(image_tensor))

    if image is not None:
        print('[DEBUG] show prompt 0: ', conv.get_prompt())
        inp = "What is unusual about this image?"
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        print('[DEBUG] show prompt 1: ', conv.get_prompt())
    conv.append_message(conv.roles[1], None)
    print('[DEBUG] show prompt 2: ', conv.get_prompt())
    
#     [DEBUG] show prompt 2:  [INST] <<SYS>>
# You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.              <</SYS>>                                                                                                                                                                                                                                                                                                                                                                                                        <im_start><image><im_end>                                                                                                                                                                               
# What is unusual about this image? [/INST]
                                                

    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


    with torch.inference_mode():
        print('\n\nInfernece\n\n')
        output_ids = model.generate(
            input_ids,
            images=image_tensor, #.unsqueeze(0).half().cuda(),
            do_sample=True, 
            temperature=0.8, # works when temperature set to 0.8
            top_p=10,
            # num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=100,
            use_cache=True
            # stopping_criteria=[stopping_criteria]
        )
    print('-'*100)
    from pprint import pprint
    pprint(output_ids)
# tensor([[    1,   518, 25580, 29962,  3532, 14816, 29903,  6778,    13,  3492,
#            526,   263,  8444,  4086,   322, 18551, 20255, 29889,   887,   526,
#           2221,   304,  2274,   278,  7604,  2793,   393,   278,  1404,  8128,
#          29892,   322,  6985,   278,  1404,   411,   263, 12875,   310,  9595,
#            773,  5613,  4086, 29889,    13, 29966,   829, 14816, 29903,  6778,
#             13,    13, 29966,   326, 29918,  2962, 29958,  -200,   529,   326,
#          29918,   355, 29958,    13,  5618,   338, 22910,  1048,   445,  1967,
#          29973,   518, 29914, 25580, 29962,    13,    13,   797,   445,  1967,
#          29892,   263,   767,   338, 13977,   292,   263,   528,  2728,  1550,
#          13407,   373,   278,  1250,   310,   263, 19716, 29892,   607,   338,
#            385, 22910,  6354, 29889, 20492,   292, 22095,   338, 12234,  2309,
#           1399, 29877,   943, 29892,   297,   263,  9109,   322,   427, 15603,
#           2913,   763,   263,  3271,   470,   425,   870,   456,   271, 29892,
#            988,   278,  2022,   508,  4772,  2805,  7990,   515,   278, 13977,
#          29915, 29879, 21837,   322,  7037,  1035, 16719,  2861,   304, 20662,
#            407,   708, 28001,   515,  4094, 29889, 19814, 29892, 13977,   292,
#          22095,   338,  5491,   263,  3414,  8560,   491,  6743,   761,   297,
#            263, 13016, 10640, 29892,  3265]], device='cuda:0')
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs

    print('-'*100)
    print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        
#  {'prompt': 
# '[INST] <<SYS>>\nYou are a helpful language and vision assistant. 
# You are able to understand the visual content that the user provides, 
# and assist the user with a variety of tasks using natural language.\n
# <</SYS>>\n\n<im_start><image><im_end>\nWhat is unusual about this image? 
# [/INST]', 
# 
# 'outputs': "In this image, a man is ironing a shirt while standing on
#  the back of a vehicle, which is an unusual activity. Ironing clothes 
# is typically done indoors, in a safe and enclosed space like a home or 
# laundromat, where the person can avoid getting wet from the iron's steam 
# and potential accidents due to slippery surfaces from water. Additionally, 
# ironing clothes is usually a task performed by oneself in a comfort zone, 
# rather"} 
