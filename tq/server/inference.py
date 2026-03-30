from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tq.core.turbo import TQConfig


@dataclass
class GenerationMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    kv_cache_size_mb: float = 0.0
    generation_time_s: float = 0.0


class InferenceEngine:
    def __init__(
        self, model_name: str, tq_config: TQConfig | None = None, device: str = "auto"
    ) -> None:
        self.model_name = model_name
        self.tq_config = tq_config
        self.device = device
        self.model = None
        self.tokenizer = None
        self._tq_cache = None
        self._total_tokens = 0
        self._start_time = time.time()

    def load(self) -> None:
        device_map = self.device if self.device != "auto" else "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=device_map,
        )
        self.model.eval()

        if self.tq_config and self.model is not None:
            self._apply_turboquant()

    def _apply_turboquant(self) -> None:
        try:
            from turboquant import TurboQuantCache

            self._tq_cache = TurboQuantCache(
                key_bits=self.tq_config.key_bits,
                value_bits=self.tq_config.value_bits,
            )
        except ImportError:
            pass

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> tuple[str, GenerationMetrics]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        if self.model.device.type != "cpu":
            input_ids = input_ids.to(self.model.device)

        prompt_tokens = input_ids.shape[1]
        start = time.time()

        if stream:
            return self._generate_stream(
                input_ids, max_tokens, temperature, top_p, prompt_tokens, start
            )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        completion_ids = outputs[0][prompt_tokens:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        elapsed = time.time() - start
        completion_tokens = len(completion_ids)

        metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=completion_tokens / elapsed if elapsed > 0 else 0,
            kv_cache_size_mb=self._estimate_kv_size(prompt_tokens + completion_tokens),
            generation_time_s=elapsed,
        )
        self._total_tokens += completion_tokens

        return text, metrics

    def _generate_stream(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        prompt_tokens: int,
        start_time: float,
    ) -> tuple[Generator[str, None, None], GenerationMetrics]:
        raise NotImplementedError("Use generate_stream instead")

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        if self.model.device.type != "cpu":
            input_ids = input_ids.to(self.model.device)

        generated = input_ids
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(generated)
                next_token_logits = outputs.logits[:, -1, :]

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_probs[sorted_indices_to_remove] = 0
                        probs = sorted_probs / sorted_probs.sum()
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=-1)
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                self._total_tokens += 1

                if next_token[0].item() == self.tokenizer.eos_token_id:
                    break

                yield token_text

    def _estimate_kv_size(self, total_tokens: int) -> float:
        if self.model is None:
            return 0.0
        config = self.model.config
        num_layers = getattr(config, "num_hidden_layers", 32)
        num_kv_heads = getattr(config, "num_key_value_heads", 8)
        head_dim = getattr(config, "hidden_size", 4096) // getattr(
            config, "num_attention_heads", 32
        )
        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2
        total_bytes = bytes_per_token * total_tokens
        if self.tq_config:
            total_bytes /= self.tq_config.compression_ratio
        return total_bytes / (1024 * 1024)

    def get_status(self) -> dict:
        uptime = time.time() - self._start_time
        return {
            "model": self.model_name,
            "turboquant": {
                "enabled": self._tq_cache is not None,
                "key_bits": self.tq_config.key_bits if self.tq_config else None,
                "value_bits": self.tq_config.value_bits if self.tq_config else None,
                "compression_ratio": self.tq_config.compression_ratio if self.tq_config else None,
            },
            "performance": {
                "tokens_generated": self._total_tokens,
                "uptime_seconds": round(uptime, 1),
            },
        }
