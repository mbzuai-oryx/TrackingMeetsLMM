# LLaMA-Adapter-V2 Multi-modal

## Setup

* setup up a new conda env and install necessary packages.
  ```bash
  conda create -n llama_adapter_v2 python=3.8 -y
  pip install -r requirements.txt
  ```

* Obtain the LLaMA backbone weights using [this form](https://forms.gle/jk851eBVbX1m5TAv5). Please note that checkpoints from unofficial sources (e.g., BitTorrent) may contain malicious code and should be used with care. Organize the downloaded file in the following structure
  ```
  /path/to/llama_model_weights
  ├── 7B
  │   ├── checklist.chk
  │   ├── consolidated.00.pth
  │   └── params.json
  └── tokenizer.model
  ```

## Pre-traininig & Fine-tuning
See [train.md](docs/train.md)
