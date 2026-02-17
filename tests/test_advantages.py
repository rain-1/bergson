from datasets import Dataset

from bergson.config import DataConfig
from bergson.utils.worker_utils import estimate_advantage


def make_reward_dataset(prompts: list[str], rewards: list[float]) -> Dataset:
    """Create a simple dataset with prompt and reward columns."""
    return Dataset.from_dict({"prompt": prompts, "reward": rewards})


def make_data_config(
    prompt_column: str = "prompt", reward_column: str = "reward"
) -> DataConfig:
    return DataConfig(
        dataset="",
        prompt_column=prompt_column,
        reward_column=reward_column,
    )


def test_estimate_advantage_single_group():
    """Advantages within a single prompt group sum to zero."""
    ds = make_reward_dataset(
        prompts=["p1", "p1", "p1"],
        rewards=[1.0, 3.0, 5.0],
    )
    cfg = make_data_config()
    advantages = estimate_advantage(ds, cfg)

    # mean = 3.0; advantages = [-2.0, 0.0, 2.0]
    assert len(advantages) == 3
    assert abs(advantages[0] - (-2.0)) < 1e-6
    assert abs(advantages[1] - 0.0) < 1e-6
    assert abs(advantages[2] - 2.0) < 1e-6
    assert abs(sum(advantages)) < 1e-6  # advantages sum to zero within group


def test_estimate_advantage_multiple_groups():
    """Advantages are computed per prompt group, not globally."""
    ds = make_reward_dataset(
        prompts=["p1", "p1", "p2", "p2"],
        rewards=[2.0, 4.0, 10.0, 20.0],
    )
    cfg = make_data_config()
    advantages = estimate_advantage(ds, cfg)

    # Group p1: mean=3.0 → advantages: [-1.0, 1.0]
    # Group p2: mean=15.0 → advantages: [-5.0, 5.0]
    assert abs(advantages[0] - (-1.0)) < 1e-6
    assert abs(advantages[1] - 1.0) < 1e-6
    assert abs(advantages[2] - (-5.0)) < 1e-6
    assert abs(advantages[3] - 5.0) < 1e-6


def test_advantages_computed_before_column_removal():
    """Regression test: advantages must be computed before reward column is dropped.

    Previously, when drop_columns=True, the reward_column was dropped before
    estimate_advantage() was called, causing a KeyError crash. This test verifies
    that advantages are correctly computed from the reward column, and that the
    reward column can be safely removed afterwards without affecting the computed
    advantages.
    """
    prompts = ["p1", "p1", "p2", "p2"]
    rewards = [1.0, 3.0, 5.0, 7.0]

    ds = make_reward_dataset(prompts=prompts, rewards=rewards)
    cfg = make_data_config()

    # Compute advantages while reward column is still present
    advantages = estimate_advantage(ds, cfg)

    # Simulate what drop_columns does: remove the reward column after advantage computation
    ds_after_drop = ds.remove_columns(["reward"])

    # The advantage column should now be added to the stripped dataset
    ds_with_advantage = ds_after_drop.add_column("advantage", advantages)

    assert "advantage" in ds_with_advantage.column_names
    assert "reward" not in ds_with_advantage.column_names

    # Group p1: mean=2.0 → advantages: [-1.0, 1.0]
    # Group p2: mean=6.0 → advantages: [-1.0, 1.0]
    expected = [-1.0, 1.0, -1.0, 1.0]
    for got, exp in zip(ds_with_advantage["advantage"], expected):
        assert abs(got - exp) < 1e-6
