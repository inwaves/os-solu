# Open-source Softmax Linear Unit

Replicating the results in the paper [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html) published recently by Anthropic. 

## Usage
Run this from the command line, using:
```bash
python3 main.py [args] 
```

The command-line arguments have reasonable defaults, so you should be able to just run the file by itself. For a full list, check out `main.py`.

## To-dos & nice-to-haves
Next up:
- profile forward pass to debug slowness
- implement layernorm with normalisation and then with scale & shift
- add functionality to specify runs via config file
- add hooks for layers
- update readme with better usage information

Nice-to-haves:
- rotary self-attention
- support for distributed training
- GUI for interaction/visualisation (probably Pysvelte-based)
- LR schedules: warmup & decay
---
license: mit
---
