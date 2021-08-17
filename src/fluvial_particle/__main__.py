"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Fluvial Particle."""


if __name__ == "__main__":
    main(prog_name="fluvial-particle")  # pragma: no cover
