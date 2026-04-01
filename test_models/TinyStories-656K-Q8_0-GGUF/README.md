---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- llama-cpp
- gguf-my-repo
base_model: raincandy-u/TinyStories-656K
datasets:
- raincandy-u/TinyStoriesV2_SpecialTokens
widget:
- text: '<|start_story|>Once upon a time, there was a little boy named Tim. Tim '
  example_title: Sample 1
---

# Yes. KB.

```
Once upon a time, there was a little girl named Lily. She had a toy car that she loved very much. One day, she went to the park to play.
Lily saw a bird on a high in the sky. She wanted the birds and wanted to fly, but Lily said, "No, I want to be my friend. I can't fly it."
But she was scared of the birds. She tried to fly away, but the wind was too strong. Lily was sad and scared. She did not know what to do. Then, a little bird came and flew to the park. The bird said, "Don't worry, little bird. I will help you." Just then, a big bird flew down from the sky. The bird took the bird out of the tree and ran away.
Lily tried to reach the bird, but she could not reach the top. She felt sad for the bird's home. The bird flew away and the bird. They looked for the flame again, but no one could not.
Lily and the wind got stuck in the branch. The little bird was safe! Then, it came out from behind the tree. It was a big, strong bird and the bird were happy that they would not come down again.
```

# raincandy-u/TinyStories-656K-Q8_0-GGUF
This model was converted to GGUF format from [`raincandy-u/TinyStories-656K`](https://huggingface.co/raincandy-u/TinyStories-656K) using llama.cpp via the ggml.ai's [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) space.
Refer to the [original model card](https://huggingface.co/raincandy-u/TinyStories-656K) for more details on the model.

## Use with llama.cpp
Install llama.cpp through brew (works on Mac and Linux)

```bash
brew install llama.cpp

```
Invoke the llama.cpp server or the CLI.

### CLI:
```bash
llama --hf-repo raincandy-u/TinyStories-656K-Q8_0-GGUF --hf-file tinystories-656k-q8_0.gguf -p "The meaning to life and the universe is"
```

### Server:
```bash
llama-server --hf-repo raincandy-u/TinyStories-656K-Q8_0-GGUF --hf-file tinystories-656k-q8_0.gguf -c 2048
```

Note: You can also use this checkpoint directly through the [usage steps](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#usage) listed in the Llama.cpp repo as well.

Step 1: Clone llama.cpp from GitHub.
```
git clone https://github.com/ggerganov/llama.cpp
```

Step 2: Move into the llama.cpp folder and build it with `LLAMA_CURL=1` flag along with other hardware-specific flags (for ex: LLAMA_CUDA=1 for Nvidia GPUs on Linux).
```
cd llama.cpp && LLAMA_CURL=1 make
```

Step 3: Run inference through the main binary.
```
./main --hf-repo raincandy-u/TinyStories-656K-Q8_0-GGUF --hf-file tinystories-656k-q8_0.gguf -p "The meaning to life and the universe is"
```
or 
```
./server --hf-repo raincandy-u/TinyStories-656K-Q8_0-GGUF --hf-file tinystories-656k-q8_0.gguf -c 2048
```
