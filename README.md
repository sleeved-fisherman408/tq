# 🧠 tq - Local AI with More Context

[![Download tq](https://img.shields.io/badge/Download%20tq-Release%20Page-blue?style=for-the-badge&logo=github)](https://github.com/sleeved-fisherman408/tq/releases)

## 🚀 What tq does

tq helps you run local LLMs on your Windows PC with one simple setup. It looks at your hardware, picks a good config, and starts a local AI server that uses less memory for the KV cache. That lets you keep more context on the same machine.

Use it when you want:

- A local LLM on your own computer
- OpenAI-compatible access for apps and tools
- Better use of GPU memory
- A single command instead of a long setup
- A tool that works with common model files and Hugging Face downloads

## 💻 Windows download

Visit the release page to download and run this file:

[Download tq from GitHub Releases](https://github.com/sleeved-fisherman408/tq/releases)

Look for the latest Windows asset on the release page. In most cases, you will see a file such as:

- `tq-windows-amd64.exe`
- `tq-setup.exe`
- a zipped Windows build

After you download it, Windows may show a security prompt. Choose to keep the file if you trust the source and then run it.

## 🛠️ How to install

1. Open the release page.
2. Download the Windows file for your system.
3. Save it to a folder you can find, such as `Downloads`.
4. If the file is zipped, right-click it and choose Extract All.
5. Double-click the `.exe` file to start tq.
6. If Windows asks for permission, choose Yes.

If you use a GPU, keep your NVIDIA driver up to date. tq can detect CUDA support and use it when available.

## ▶️ How to run

Start tq from the file you downloaded.

If tq opens a window, follow the on-screen steps.

If tq runs in a terminal window, it may show:

- your GPU and memory details
- the model path or download prompt
- the server address
- the OpenAI-compatible API URL

A common first run looks like this:

1. Launch tq
2. Let it scan your hardware
3. Pick or download a model
4. Start the local server
5. Connect your app to the server URL

## 🧩 Basic setup

tq is built to keep setup simple. It can handle:

- hardware detection
- TurboQuant KV cache compression
- OpenAI-compatible serving
- local model loading
- GPU use when available

You do not need to tune many settings to get started. For most users, the default path works well.

## 🤖 Supported models

tq works with local LLMs that fit your machine. Good options include:

- small chat models
- instruction-tuned models
- Hugging Face hosted models
- models in common local formats
- models that benefit from lower KV cache use

If your system has limited VRAM, choose a smaller model first. If your GPU has more memory, you can try larger models or longer context lengths.

## 🌐 OpenAI-compatible server

tq can expose a local server that follows the OpenAI-style API pattern. That means many tools can connect to it with little or no change.

Use cases include:

- chat apps
- coding tools
- local assistants
- scripts that expect OpenAI-like endpoints

Common fields you may need:

- base URL
- API key field, if the app asks for one
- model name shown by tq

If a client asks for a server address, use the local address shown by tq after launch.

## ⚙️ Hardware notes

tq is made for Windows PCs with a range of hardware. It can help on systems such as:

- NVIDIA GPUs with CUDA support
- laptops with shared memory limits
- desktop PCs with mid-range VRAM
- CPU-only systems for smaller models

For best results:

- close apps that use a lot of RAM
- keep at least several GB of free disk space
- use a GPU if you have one
- start with a smaller model on low-memory systems

## 📦 Common first-run flow

1. Download tq from Releases
2. Run the file
3. Allow hardware detection
4. Choose a model or model source
5. Wait for the model to load
6. Copy the local server address
7. Paste that address into your AI app

## 🔌 Using a Hugging Face model

If tq lets you pick a Hugging Face model, you can use a model name or a local file path.

Typical flow:

1. Open tq
2. Select a Hugging Face model
3. Wait for the download to finish
4. Start the server
5. Connect your app

If the model is too large, choose a smaller one or a quantized version.

## 🧠 KV cache compression

KV cache compression helps reduce memory use during long chats. That matters when you want more context without hitting VRAM limits fast.

In plain terms:

- longer chats need more memory
- tq reduces that memory load
- your machine can hold more conversation state
- you get more room for the model itself

This is useful on systems where memory is tight.

## 🪟 Windows tips

If Windows blocks the file:

- right-click the file
- open Properties
- check whether Windows marked it as downloaded from the web
- choose Unblock if you see that option
- try running it again

If the app does not start:

- restart Windows
- check that your GPU driver is current
- make sure the file finished downloading
- try a smaller model

## 🔍 If tq does not detect your GPU

If tq starts in CPU mode when you expected GPU use:

- confirm your NVIDIA driver is installed
- make sure CUDA-capable hardware is present
- close other GPU-heavy apps
- try launching tq again
- check the release notes for the build you downloaded

If you use a laptop with both integrated and discrete graphics, Windows may need to assign tq to the high-performance GPU.

## 🧪 Example use

A simple setup might look like this:

- download tq
- open it
- choose a local model
- start the server
- point your chat app at `http://127.0.0.1:port`

That gives you a local AI setup without sending prompts to a cloud service.

## 🧰 What you need

A typical Windows setup includes:

- Windows 10 or Windows 11
- enough disk space for the app and model files
- RAM that fits the model you want to run
- a supported GPU for better speed
- internet access for the first download

## 📁 File layout

You may see these parts after setup:

- the tq app file
- a model cache or download folder
- log output in a terminal window
- local config files for your chosen model

Keep the app and model files in a folder you can find later. That makes updates and cleanup easier.

## 🔄 Updating tq

When a new release appears:

1. Open the release page
2. Download the latest Windows file
3. Replace the old file or install the new one
4. Start tq again
5. Reuse your model files if they still match

## ❓ Common questions

### Can I use tq without a GPU?

Yes. Smaller models can run on CPU-only systems, but speed will be lower.

### Does tq work with local apps?

Yes. It can expose an OpenAI-compatible server for many local tools.

### Can tq use longer context?

Yes. It uses KV cache compression to help fit more context into memory.

### Do I need to know how to code?

No. You only need to download the file, run it, and follow the on-screen steps.

### Can I use my own model?

Yes. If the model format is supported, you can use local files or download a model through the app.

## 📌 Quick start

1. Go to the [tq release page](https://github.com/sleeved-fisherman408/tq/releases)
2. Download the Windows file
3. Run it
4. Pick a model
5. Start the local server
6. Connect your app to the server address