"""Numerical, causal, cache, export, and objective contracts for RealSemi."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from transformers import Qwen3Config, AutoModel, Qwen3ForCausalLM
from litespec.modeling.draft.realsemi import RealSemiDraftModel
from litespec.modeling.draft.dflash import DFlashDraftModel
from litespec.core.attention_masks import (
    create_dflash_sdpa_mask,
    create_dflash_flex_mask,
)
from litespec.core.realsemi import OnlineRealSemiModel
from litespec.training.checkpoint import export_draft


def config(mode="full", n=4, m=2, backend="sdpa"):
    c = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        attention_dropout=0.0,
    )
    c.num_target_layers = 3
    c.architectures = ["RealSemiDraftModel"]
    c._attn_implementation = backend
    c.block_size = 4
    c.layer_types = ["full_attention"] * n
    c.sliding_window = None
    c.use_sliding_window = False
    c.realsemi_config = dict(
        num_parallel_layers=m,
        serial_context_mode=mode,
        target_layer_ids=[0, 1],
        mask_token_id=63,
        attention_mode="gqa",
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
    )
    return c


def fixture(model, device="cpu"):
    batch, seq, blocks = 2, 12, 2
    anchors = torch.tensor([[2, 6], [3, 7]], device=device)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    context = torch.randn(batch, seq, 64, device=device)
    noise = torch.randn(batch, blocks * 4, 32, device=device)
    previous = torch.randn(batch, blocks, 3, 32, device=device)
    positions = torch.cat(
        (
            torch.arange(seq, device=device)[None].expand(batch, -1),
            (anchors[..., None] + torch.arange(4, device=device)).reshape(batch, -1),
        ),
        -1,
    )
    builder = (
        create_dflash_flex_mask
        if model.config._attn_implementation == "flex_attention"
        else create_dflash_sdpa_mask
    )
    mask = builder(anchors, keep, seq, 4, device)
    args = dict(
        noise_embedding=noise,
        target_hidden=context,
        position_ids=positions,
        previous_token_embeddings=previous,
        anchor_positions=anchors,
        block_keep_mask=keep,
        attention_mask=mask,
    )
    return args


class RealSemiTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(2)

    def test_block_matches_cached_steps_and_cache_lengths(self):
        for mode in ("full", "last"):
            for n, m in ((5, 4), (4, 2)):
                model = RealSemiDraftModel(config(mode, n, m)).eval()
                args = fixture(model)
                with torch.no_grad():
                    reference = model(**args)
                    z = model.forward_parallel(
                        noise_embedding=args["noise_embedding"],
                        target_hidden=args["target_hidden"],
                        position_ids=args["position_ids"],
                        attention_mask=args["attention_mask"],
                    ).reshape(2, 2, 4, 32)
                    for b in range(2):
                        for block in range(2):
                            a = int(args["anchor_positions"][b, block])
                            cache = model.prepare_serial_cache(
                                args["target_hidden"][b : b + 1, :a],
                                torch.arange(a)[None],
                            )
                            for i in range(3):
                                result = model.serial_step(
                                    args["previous_token_embeddings"][
                                        b : b + 1, block, i : i + 1
                                    ],
                                    z[b : b + 1, block, i + 1 : i + 2],
                                    torch.tensor([[a + i + 1]]),
                                    cache,
                                )
                                torch.testing.assert_close(
                                    result[:, 0],
                                    reference[b : b + 1, block, i],
                                    atol=2e-5,
                                    rtol=2e-5,
                                )
                                for layer in range(m, n):
                                    self.assertEqual(
                                        cache.context[layer][0].size(-2),
                                        a if mode == "full" else 1,
                                    )
                                    self.assertEqual(
                                        cache.draft[layer][0].size(-2), i + 1
                                    )
                            state = dict(cache.context)
                            next_cache = model.prepare_serial_cache(
                                args["target_hidden"][b : b + 1, a : a + 2],
                                torch.arange(a, a + 2)[None],
                                state,
                            )
                            self.assertFalse(next_cache.draft)
                            self.assertEqual(
                                next_cache.context[m][0].size(-2),
                                a + 2 if mode == "full" else 1,
                            )

    def test_causal_and_anchor_isolation(self):
        for mode in ("full", "last"):
            model = RealSemiDraftModel(config(mode)).eval()
            args = fixture(model)
            with torch.no_grad():
                original = model(**args)
                changed = dict(args)
                changed["previous_token_embeddings"] = args[
                    "previous_token_embeddings"
                ].clone()
                changed["previous_token_embeddings"][:, :, -1] += 9
                torch.testing.assert_close(
                    model(**changed)[:, :, :-1], original[:, :, :-1]
                )
                changed = dict(args)
                changed["noise_embedding"] = args["noise_embedding"].clone()
                changed["noise_embedding"][:, 4:] += 9
                torch.testing.assert_close(model(**changed)[:, 0], original[:, 0])
                changed = dict(args)
                changed["target_hidden"] = args["target_hidden"].clone()
                for b in range(2):
                    changed["target_hidden"][
                        b, int(args["anchor_positions"][b, 0]) :
                    ] += 7
                torch.testing.assert_close(model(**changed)[:, 0], original[:, 0])
                changed = dict(args)
                changed["block_keep_mask"] = args["block_keep_mask"].clone()
                changed["block_keep_mask"][:, 1] = False
                self.assertTrue(torch.isfinite(model(**changed)).all())

    def test_loss_gradient_chunking_and_alignment(self):
        model = RealSemiDraftModel(config())
        embed, head = nn.Embedding(64, 32), nn.Linear(32, 64, bias=False)
        online = OnlineRealSemiModel(
            model,
            head,
            embed,
            63,
            block_size=4,
            attention_backend="sdpa",
            num_anchors=2,
            objective_chunk_blocks=0,
        )
        ids = torch.randint(0, 60, (2, 12))
        context, teacher = torch.randn(2, 12, 64), torch.randn(2, 12, 32)
        mask = torch.ones(2, 12)
        mask[:, 10:] = 0
        seed = torch.random.get_rng_state()
        captured = {}

        def capture(module, args, kwargs, result):
            captured.update(inputs=kwargs, predictions=result)

        hook = model.register_forward_hook(capture, with_kwargs=True)
        out = online(ids, context, mask, teacher)
        hook.remove()
        anchors = captured["inputs"]["anchor_positions"]
        index = anchors[..., None] + torch.arange(1, 4)
        safe = index.clamp(max=11)
        gold = ids[:, None].expand(-1, 2, -1).gather(2, safe)
        weight = (index < 12).float() * mask[:, None].expand(-1, 2, -1).gather(2, safe)
        logits = head(captured["predictions"]).float()
        teacher_states = online._aligned_target_hidden(teacher, safe)
        target_logits = head(teacher_states).detach().float()
        ce_reference = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 64), gold.flatten(), reduction="none"
        ).reshape_as(gold)
        l1_reference = (logits.softmax(-1) - target_logits.softmax(-1)).abs().sum(-1)
        manual = (
            (0.1 * ce_reference + 0.9 * l1_reference) * weight
        ).sum() / weight.sum()
        torch.testing.assert_close(out.loss, manual)
        from litespec.core.block_metrics import (
            block_accepted_length_terms, expected_accept_len_terms,
            distribution_acceptance_probability,
        )
        from litespec.training.logging import build_training_log_payload
        walk = block_accepted_length_terms(logits.argmax(-1), gold, weight > 0)
        expected = expected_accept_len_terms(
            distribution_acceptance_probability(target_logits, logits), weight > 0,
        )
        for key, reference in (("hard_label/block_accepted_length", walk),
                               ("expected_accept_len", expected)):
            actual = out.ratio_metrics[key]
            torch.testing.assert_close(actual.numerator, reference.numerator)
            torch.testing.assert_close(actual.denominator, reference.denominator, check_dtype=False)
        reduced = {key: value.numerator / value.denominator
                   for key, value in out.ratio_metrics.items()}
        payload = build_training_log_payload(reduced, 0.001, detail_prefix="realsemi")
        self.assertIn("train/walk_accepted_length", payload)
        self.assertIn("train/expected_accept_len", payload)
        self.assertGreaterEqual(payload["train/walk_accepted_length"], 1.0)
        self.assertLessEqual(payload["train/expected_accept_len"], 4.0)

        expected = (
            0.1 * out.ratio_metrics["objective/ce"].numerator
            + 0.9 * out.ratio_metrics["objective/l1"].numerator
        )
        torch.testing.assert_close(out.loss_terms.numerator, expected)
        out.loss.backward()
        grads = {name: value.grad.clone() for name, value in model.named_parameters()}
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))
        self.assertTrue(all(g.abs().sum() > 0 for g in grads.values()))
        self.assertIsNone(head.weight.grad)
        self.assertIsNone(embed.weight.grad)
        model.zero_grad(set_to_none=True)
        online.objective_chunk_blocks = 1
        torch.random.set_rng_state(seed)
        chunked = online(ids, context, mask, teacher)
        torch.testing.assert_close(chunked.loss, out.loss)
        chunked.loss.backward()
        for name, value in model.named_parameters():
            torch.testing.assert_close(value.grad, grads[name], atol=2e-5, rtol=2e-5)
        gathered = online._aligned_target_hidden(
            teacher, torch.tensor([[[3, 4]], [[5, 6]]])
        )
        torch.testing.assert_close(gathered[0, 0], teacher[0, 2:4])
        torch.testing.assert_close(gathered[1, 0], teacher[1, 4:6])

    def test_export_and_existing_architecture(self):
        model = RealSemiDraftModel(config("last")).eval()
        args = fixture(model)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as path:
            export_draft(model, path)
            loaded = RealSemiDraftModel.from_pretrained(
                path, attn_implementation="sdpa"
            )
            torch.testing.assert_close(model(**args), loaded(**args))
            loaded = AutoModel.from_pretrained(
                path, trust_remote_code=True, attn_implementation="sdpa"
            )
            torch.testing.assert_close(model(**args), loaded(**args))
        c = config()
        settings = c.realsemi_config
        del c.realsemi_config
        c.dflash_config = {
            k: v
            for k, v in settings.items()
            if k in ("target_layer_ids", "mask_token_id", "attention_mode")
        }
        c.architectures = ["DFlashDraftModel"]
        baseline = DFlashDraftModel(c)
        args = fixture(baseline)
        hidden = baseline(
            position_ids=args["position_ids"],
            attention_mask=args["attention_mask"],
            noise_embedding=args["noise_embedding"],
            target_hidden=args["target_hidden"],
        )
        self.assertEqual(tuple(hidden.shape), (2, 8, 32))

    def test_identical_initial_weights_and_zero_anchor(self):
        torch.manual_seed(42)
        full = RealSemiDraftModel(config("full")).eval()
        torch.manual_seed(42)
        last = RealSemiDraftModel(config("last")).eval()
        for name, parameter in full.state_dict().items():
            torch.testing.assert_close(
                parameter, last.state_dict()[name], rtol=0, atol=0
            )
        for model in (full, last):
            args = fixture(model)
            args["anchor_positions"][:, 0] = 0
            args["position_ids"][:, 12:16] = torch.arange(4)
            args["attention_mask"] = create_dflash_sdpa_mask(
                args["anchor_positions"], args["block_keep_mask"], 12, 4, "cpu"
            )
            with torch.no_grad():
                original = model(**args)[:, 0]
                args["target_hidden"] = args["target_hidden"] + 5
                torch.testing.assert_close(original, model(**args)[:, 0])

    def test_speculative_matches_target_greedy(self):
        for mode in ("full", "last"):
            c = config(mode)
            model = RealSemiDraftModel(c).eval()
            tc = Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                eos_token_id=None,
                pad_token_id=0,
            )
            tc._attn_implementation = "sdpa"
            target = Qwen3ForCausalLM(tc).eval()
            inputs = torch.tensor([[1, 3, 5]])
            for count in (1, 2, 11):
                with torch.no_grad():
                    expected = target.generate(
                        inputs, max_new_tokens=count, do_sample=False
                    )
                    actual = model.spec_generate(target, inputs, count, [], 0.0)
                torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_flex_forward_backward(self):
        for mode in ("full", "last"):
            reference = RealSemiDraftModel(config(mode)).cuda()
            model = RealSemiDraftModel(config(mode, backend="flex_attention")).cuda()
            model.load_state_dict(reference.state_dict())
            torch.manual_seed(4)
            args = fixture(reference, "cuda")
            args["block_keep_mask"][:, 1] = False
            args["attention_mask"] = create_dflash_sdpa_mask(
                args["anchor_positions"], args["block_keep_mask"], 12, 4, "cuda"
            )
            expected = reference(**args)
            flexargs = dict(args)
            flexargs["attention_mask"] = create_dflash_flex_mask(
                args["anchor_positions"], args["block_keep_mask"], 12, 4, "cuda"
            )
            actual = model(**flexargs)
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
            expected.square().sum().backward()
            actual.square().sum().backward()
            for (name, p), (_, q) in zip(
                model.named_parameters(), reference.named_parameters()
            ):
                torch.testing.assert_close(
                    p.grad, q.grad, atol=3e-4, rtol=3e-4, msg=name
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
