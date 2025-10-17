"""Quick example showcasing the symmetry coupling metric."""

from symmetry_coupling import Config, compute_coupling


def main() -> None:
    config = Config(language="he", domain="general", normalize=True)
    metrics = compute_coupling(
        "האדם שם את הספר על השולחן",
        "על השולחן הנח אדם ספר",
        config,
    )

    print(metrics)
    print("Divergence:", metrics.divergence)
    print("Classification:", metrics.classification)


if __name__ == "__main__":
    main()
