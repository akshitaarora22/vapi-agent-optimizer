"""
tests/test_core.py

Basic sanity tests — run with: pytest tests/
"""

import pytest
from agent.config import build_system_prompt, BASELINE_CONFIG, PROMPT_AXES, AXIS_SIZES
from optimizer.gp_optimizer import GPBayesianOptimizer
from evaluator.scorer import _score_info_completeness


class TestPromptConfig:
    def test_baseline_config_valid(self):
        prompt = build_system_prompt(BASELINE_CONFIG)
        assert "Bright Smile Dental" in prompt
        assert len(prompt) > 200

    def test_all_axis_combinations_build(self):
        """Every valid config should build a prompt without error."""
        import itertools
        axes = list(PROMPT_AXES.keys())
        for combo in itertools.product(*[range(s) for s in AXIS_SIZES.values()]):
            config = dict(zip(axes, combo))
            prompt = build_system_prompt(config)
            assert prompt is not None

    def test_different_configs_produce_different_prompts(self):
        config_a = {k: 0 for k in PROMPT_AXES}
        config_b = {k: min(1, AXIS_SIZES[k] - 1) for k in PROMPT_AXES}
        assert build_system_prompt(config_a) != build_system_prompt(config_b)


class TestOptimizer:
    def test_vector_config_roundtrip(self):
        opt = GPBayesianOptimizer()
        config = {"greeting_style": 1, "information_gathering": 0, "error_recovery": 2, "confirmation_style": 1}
        vec = opt.config_to_vector(config)
        assert opt.vector_to_config(vec) == config

    def test_optimize_calls_objective(self):
        """GP optimizer should call objective n_iterations times."""
        opt = GPBayesianOptimizer(n_random_starts=2)
        call_count = [0]

        def fake_objective(config):
            call_count[0] += 1
            return 0.5

        opt.optimize(fake_objective, n_iterations=4)
        assert call_count[0] == 4

    def test_history_tracks_all_evals(self):
        opt = GPBayesianOptimizer(n_random_starts=2)
        opt.optimize(lambda c: 0.7, n_iterations=4)
        assert len(opt.history) == 4

    def test_best_config_is_highest_reward(self):
        opt = GPBayesianOptimizer(n_random_starts=2)
        rewards = [0.3, 0.8, 0.5, 0.6]
        idx = [0]

        def fake_objective(config):
            r = rewards[idx[0]]
            idx[0] += 1
            return r

        opt.optimize(fake_objective, n_iterations=4)
        assert opt.best_reward() == 0.8


class TestScorer:
    def test_info_completeness_full(self):
        transcript = [
            {"role": "assistant", "content": "What is your name?"},
            {"role": "user", "content": "Jane Smith"},
            {"role": "assistant", "content": "What date and time works for you?"},
            {"role": "user", "content": "Tuesday at 3pm"},
            {"role": "assistant", "content": "What service do you need? Cleaning, whitening?"},
            {"role": "user", "content": "Teeth cleaning"},
            {"role": "assistant", "content": "May I have your phone number?"},
            {"role": "user", "content": "555-1234"},
        ]
        score = _score_info_completeness(transcript)
        assert score == 1.0

    def test_info_completeness_partial(self):
        transcript = [
            {"role": "assistant", "content": "What is your name?"},
            {"role": "user", "content": "John"},
        ]
        score = _score_info_completeness(transcript)
        assert 0 < score < 1.0

    def test_info_completeness_empty(self):
        score = _score_info_completeness([])
        assert score == 0.0
