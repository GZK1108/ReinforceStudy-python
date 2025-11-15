"""GRPO training entry point extracted from grpo.ipynb.

This module converts the notebook prototype into a reusable script so we can
train a LoRA-adapted causal language model with GRPO (Generative Rewards Policy
Optimization) for Java unit test generation.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:
    import deepspeed  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    deepspeed = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError("transformers must be installed to run grpo.py") from exc

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError("peft must be installed to run grpo.py") from exc

LOGGER = logging.getLogger("grpo")


def setup_logging(verbosity: int = logging.INFO) -> None:
    """Configure root logging once."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=verbosity,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module (DeepSpeed/Peft aware)."""
    return model.module if hasattr(model, "module") else model


@dataclass
class GRPOConfig:
    base_model_path: str
    dataset_path: str
    lora_model_path: Optional[str] = None
    ref_model_path: Optional[str] = None
    output_dir: str = "outputs/grpo"
    deepspeed_config: Optional[str] = None
    local_rank: int = -1
    epochs: int = 1
    num_samples: int = 1
    beta: float = 0.1
    cliprange: float = 0.2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_prompt_tokens: int = 1024
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    seed: int = 42
    update_old_policy_every: int = 10
    log_interval: int = 10
    device: Optional[str] = None
    jacoco_path: Optional[str] = None
    pit_path: Optional[str] = None

    def resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        if self.local_rank != -1 and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            return torch.device("cuda", self.local_rank)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


class TestGenerationEnvironment:
    """Light-weight static evaluator used to shape GRPO rewards."""

    def __init__(self, jacoco_path: Optional[str] = None, pit_path: Optional[str] = None):
        self.coverage_weight = 0.4
        self.mutation_weight = 0.3
        self.readability_weight = 0.3
        self.jacoco_path = jacoco_path
        self.pit_path = pit_path
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_eval_"))
        LOGGER.info("Created temporary evaluation folder: %s", self.temp_dir)

    # --- public API -----------------------------------------------------
    def evaluate_test(
        self,
        generated_test: str,
        source_code: str,
        reference_test: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        LOGGER.debug("Evaluating generated test with %d chars", len(generated_test))
        try:
            coverage_score = self._static_coverage_analysis(generated_test, source_code)
            mutation_score = self._static_mutation_analysis(generated_test)
            readability_score = self._calculate_readability(generated_test)
            similarity_score = 0.0
            if reference_test:
                similarity_score = self._calculate_similarity(generated_test, reference_test)

            total_score = (
                self.coverage_weight * coverage_score
                + self.mutation_weight * mutation_score
                + self.readability_weight * readability_score
            )

            return total_score, {
                "coverage_score": coverage_score,
                "mutation_score": mutation_score,
                "readability_score": readability_score,
                "similarity_score": similarity_score,
            }
        except Exception as exc:  # pragma: no cover - safety net
            LOGGER.error("Evaluation failed: %s", exc)
            return 0.65, {
                "coverage_score": 0.6,
                "mutation_score": 0.6,
                "readability_score": 0.8,
                "similarity_score": 0.0,
                "error": str(exc),
            }

    def cleanup(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- heuristic scorers ----------------------------------------------
    def _static_coverage_analysis(self, generated_test: str, source_code: str) -> float:
        """Estimate coverage by matching method names present in the source."""
        method_tokens = self._extract_method_tokens(source_code)
        hits = sum(1 for token in method_tokens if token in generated_test)
        base = 0.1 if hits == 0 else hits / (len(method_tokens) + 1e-6)
        return float(max(0.0, min(1.0, base)))

    def _static_mutation_analysis(self, generated_test: str) -> float:
        mutation_indicators = ["assert", "mock", "when", "then", "verify", "throw"]
        hits = sum(1 for word in mutation_indicators if word.lower() in generated_test.lower())
        return float(min(1.0, 0.2 + hits / len(mutation_indicators)))

    def _calculate_readability(self, generated_test: str) -> float:
        lines = [line.strip() for line in generated_test.splitlines() if line.strip()]
        if not lines:
            return 0.0
        avg_len = sum(len(line) for line in lines) / len(lines)
        variance = sum((len(line) - avg_len) ** 2 for line in lines) / len(lines)
        score = math.exp(-variance / (avg_len + 1e-6))
        return float(max(0.0, min(1.0, score)))

    def _calculate_similarity(self, generated_test: str, reference_test: str) -> float:
        return float(SequenceMatcher(None, generated_test, reference_test).ratio())

    def _extract_method_tokens(self, source_code: str) -> List[str]:
        tokens: List[str] = []
        for line in source_code.splitlines():
            line = line.strip()
            if line.startswith("public") or line.startswith("private") or line.startswith("protected"):
                if "(" in line and ")" in line:
                    method_name = line.split("(")[0].split()[-1]
                    tokens.append(method_name)
        return tokens

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.cleanup()


class GRPOTrainer:
    """GRPO trainer converted from the experimental notebook."""

    def __init__(self, config: GRPOConfig):
        setup_logging()
        self.config = config
        if self.config.local_rank != -1 and deepspeed is not None:
            deepspeed.init_distributed()
        self.device = config.resolve_device()
        LOGGER.info("Using device: %s", self.device)
        self._seed_everything(config.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = self._load_model()
        self.old_model: Optional[nn.Module] = None
        self.ref_model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.using_deepspeed = False
        self.training_steps = 0

        self._setup_trainable_params()
        self._setup_optimizer()
        self._setup_reference_models()

        self.environment = TestGenerationEnvironment(config.jacoco_path, config.pit_path)
        self.train_data = self._load_dataset(config.dataset_path)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    # --- initialization helpers ----------------------------------------
    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_model(self) -> nn.Module:
        cuda_available = torch.cuda.is_available()
        dtype = torch.float16 if cuda_available else torch.float32
        model_kwargs = {"torch_dtype": dtype, "use_cache": False}

        model = AutoModelForCausalLM.from_pretrained(self.config.base_model_path, **model_kwargs)

        if self.config.lora_model_path:
            model = PeftModel.from_pretrained(model, self.config.lora_model_path)

        return model

    def _setup_trainable_params(self) -> None:
        trainable_params: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if self.config.lora_model_path:
                if "lora" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params.append(param)
        self.trainable_params = trainable_params

    def _setup_optimizer(self) -> None:
        if self.config.deepspeed_config:
            if deepspeed is None:
                raise ImportError("deepspeed is required when deepspeed_config is provided")
            LOGGER.info("Initialising DeepSpeed with config: %s", self.config.deepspeed_config)
            with open(self.config.deepspeed_config, "r", encoding="utf-8") as fp:
                ds_config = json.load(fp)
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=self.trainable_params,
                config=ds_config,
            )
            self.model = model_engine
            self.optimizer = optimizer
            self.using_deepspeed = True
        else:
            self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.using_deepspeed = False

    def _setup_reference_models(self) -> None:
        self.create_old_model_copy(force_new=True)
        if self.config.ref_model_path:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model_kwargs = {"torch_dtype": dtype, "use_cache": False}
            ref_model = AutoModelForCausalLM.from_pretrained(self.config.ref_model_path, **model_kwargs)
            ref_model.to(self.device)
            ref_model.eval()
            self.ref_model = ref_model

    def _load_dataset(self, dataset_path: str) -> List[Dict[str, str]]:
        with open(dataset_path, "r", encoding="utf-8") as fp:
            json_data = json.load(fp)
        if isinstance(json_data, dict) and "data" in json_data:
            raw_samples = json_data["data"]
        elif isinstance(json_data, list):
            raw_samples = json_data
        else:
            raise ValueError("Unsupported dataset schema. Expect list or dict with 'data'.")

        train_data: List[Dict[str, str]] = []
        for item in raw_samples:
            source_code = item.get("source_code", "")
            test_code = item.get("test_code", "")
            prompt = item.get("prompt")
            if not prompt:
                prompt = f"请为以下Java类生成单元测试用例: ```java\n{source_code}\n```\n生成的测试用例："
            train_data.append(
                {
                    "prompt": prompt,
                    "source_code": source_code,
                    "reference_test": test_code,
                }
            )
        LOGGER.info("Loaded %d training samples", len(train_data))
        return train_data

    # --- training loop --------------------------------------------------
    def train(self) -> Dict[str, float]:
        if not self.train_data:
            LOGGER.warning("No training data found. Exiting.")
            return {}

        aggregate_stats = {
            "loss": 0.0,
            "mean_reward": 0.0,
            "mean_kl_div": 0.0,
            "num_steps": 0,
        }
        effective_epochs = 0

        for epoch in range(self.config.epochs):
            random.shuffle(self.train_data)
            total_loss = 0.0
            total_reward = 0.0
            total_kl = 0.0
            total_samples = 0

            for idx, sample in enumerate(self.train_data):
                metrics = self._train_single_sample(sample)
                if metrics is None:
                    continue
                total_loss += metrics["loss"]
                total_reward += metrics["reward"]
                total_kl += metrics["kl_div"]
                total_samples += 1

                if total_samples % self.config.log_interval == 0:
                    LOGGER.info(
                        "Epoch %d | Sample %d | loss=%.4f | reward=%.4f | kl=%.4f",
                        epoch + 1,
                        total_samples,
                        metrics["loss"],
                        metrics["reward"],
                        metrics["kl_div"],
                    )

                if total_samples % self.config.update_old_policy_every == 0:
                    LOGGER.info("Sync old policy after %d samples", total_samples)
                    self.create_old_model_copy()

            if total_samples:
                epoch_loss = total_loss / total_samples
                epoch_reward = total_reward / total_samples
                epoch_kl = total_kl / total_samples
                LOGGER.info(
                    "Epoch %d finished | loss=%.4f | reward=%.4f | kl=%.4f",
                    epoch + 1,
                    epoch_loss,
                    epoch_reward,
                    epoch_kl,
                )
                aggregate_stats["loss"] += epoch_loss
                aggregate_stats["mean_reward"] += epoch_reward
                aggregate_stats["mean_kl_div"] += epoch_kl
                aggregate_stats["num_steps"] += total_samples
                effective_epochs += 1

        if effective_epochs:
            aggregate_stats["loss"] /= effective_epochs
            aggregate_stats["mean_reward"] /= effective_epochs
            aggregate_stats["mean_kl_div"] /= effective_epochs

        self._save_training_summary(aggregate_stats)
        return aggregate_stats

    def _train_single_sample(self, sample: Dict[str, str]) -> Optional[Dict[str, float]]:
        prompt = sample["prompt"]
        source_code = sample.get("source_code", "")
        reference_test = sample.get("reference_test")

        try:
            generated_tests = self.generate_test(prompt, self.config.num_samples)
        except Exception as exc:
            LOGGER.error("Generation failed: %s", exc)
            return None

        rewards, metrics_list = self.compute_rewards(generated_tests, source_code, reference_test)
        if rewards.numel() == 0:
            LOGGER.warning("Rewards empty, skipping sample")
            return None

        value = rewards.mean()
        advantages = rewards - value
        if advantages.numel() > 1 and torch.std(advantages) > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_texts = [f"{prompt}\n{test}" for test in generated_tests]
        tokenized = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_tokens,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        current_log_probs = self.compute_logprobs(self.model, input_ids, attention_mask)
        with torch.no_grad():
            old_log_probs = self.compute_logprobs(self.old_model, input_ids, attention_mask)

        ratios = torch.exp(current_log_probs - old_log_probs)
        clipped_ratios = torch.clamp(ratios, 1 - self.config.cliprange, 1 + self.config.cliprange)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        kl_div = self.calculate_kl_divergence(input_ids, attention_mask)
        loss = policy_loss + self.config.beta * kl_div

        if self.using_deepspeed:
            self.model.backward(loss)
            self.model.step()
        else:
            assert self.optimizer is not None
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.max_grad_norm)
            self.optimizer.step()

        self.training_steps += 1
        avg_reward = rewards.mean().item()
        return {
            "loss": loss.item(),
            "reward": avg_reward,
            "kl_div": kl_div.item(),
            "metrics": metrics_list,
        }

    # --- objective helpers ---------------------------------------------
    def generate_test(self, prompt: str, num_samples: int) -> List[str]:
        model = unwrap_model(self.model)
        prev_mode = model.training
        model.eval()
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_tokens,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generations = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                num_return_sequences=num_samples,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        decoded = self.tokenizer.batch_decode(
            generations[:, prompt_length:], skip_special_tokens=True
        )
        self.model.train(prev_mode)
        return decoded

    def compute_rewards(
        self,
        generated_tests: Sequence[str],
        source_code: str,
        reference_test: Optional[str],
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        rewards = []
        metrics_list = []
        for test in generated_tests:
            reward, metrics = self.environment.evaluate_test(test, source_code, reference_test)
            rewards.append(reward)
            metrics_list.append(metrics)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        return reward_tensor, metrics_list

    def compute_logprobs(
        self,
        model: Optional[nn.Module],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if model is None:
            raise ValueError("Model must be initialised before computing log probs")
        module = unwrap_model(model)
        outputs = module(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask = attention_mask[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * mask
        return token_log_probs.sum(dim=1)

    def calculate_kl_divergence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        current_log_probs = self.compute_logprobs(self.model, input_ids, attention_mask)
        if self.ref_model is not None:
            with torch.no_grad():
                ref_log_probs = self.compute_logprobs(self.ref_model, input_ids, attention_mask)
        else:
            with torch.no_grad():
                ref_log_probs = self.compute_logprobs(self.old_model, input_ids, attention_mask)
        kl = current_log_probs - ref_log_probs
        return kl.mean()

    def create_old_model_copy(self, force_new: bool = False) -> None:
        base_model = unwrap_model(self.model)
        if self.old_model is None or force_new:
            self.old_model = copy.deepcopy(base_model).to(self.device)
        else:
            self.old_model.load_state_dict(base_model.state_dict())
        for param in self.old_model.parameters():
            param.requires_grad = False
        self.old_model.eval()

    def _save_training_summary(self, stats: Dict[str, float]) -> None:
        summary_path = self.output_dir / "training_summary.json"
        payload = {**stats, "config": asdict(self.config)}
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        LOGGER.info("Saved training summary to %s", summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO fine-tuning")
    parser.add_argument("--base_model_path", type=str, required=True, help="Base HF model path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Training dataset JSON path")
    parser.add_argument("--lora_model_path", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--ref_model_path", type=str, default=None, help="Reference model for KL term")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo", help="Directory for logs and checkpoints")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed JSON config")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed setup")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty weight")
    parser.add_argument("--cliprange", type=float, default=0.2, help="PPO clipping range")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--max_prompt_tokens", type=int, default=1024, help="Tokenizer max length")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Generation max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling p")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--update_old_policy_every", type=int, default=10, help="How often to sync old policy")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0")
    parser.add_argument("--jacoco_path", type=str, default=None, help="Optional JaCoCo path for future integration")
    parser.add_argument("--pit_path", type=str, default=None, help="Optional PIT path for future integration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GRPOConfig(**vars(args))
    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
