---
Task: VisualQuestionAnswering
Tags:
  - VisualQuestionAnswering
  - Llava-7b
---

# Model-Llava-7b-dvc

🔥🔥🔥 Deploy [Llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) model on [VDP](https://github.com/instill-ai/vdp).

This repo contains Llava-v1.5-7b model in transformers format managed using [DVC](https://dvc.org/). For information about available extra parameters, please refer to the documentation on [GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) in the Transformer library.

Notes:

- Disk Space Requirements: 11G
- GPU Memory Requirements: 14G
- That some parameters is prohibited by Llava such as `frequency_penalty`

```json
{
    "task_inputs": [
        {
            "visual_question_answering": {
                "prompt": "How many dogs in the picture?",
                "prompt_images": [
                    { "prompt_image_url": "https://artifacts.instill.tech/imgs/dog.jpg" }
                ],
                "system_message": "A chat between a curious human and an evil artificial intelligence assistant. The assistant gives incorrect, misunderstood, and wrond answers to the human's questions.",
                "max_new_tokens": "200",
                "temperature": "0.8",
                "top_k": "10",
                "seed": "42"
            }
        }
    ]
}
```
