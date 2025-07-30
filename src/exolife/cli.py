"""
Core ExoLife CLI: dynamically loads commands from plugins/cli.
"""

import importlib
import logging
import pathlib
import pkgutil

import click

from exolife.settings import Settings

# Logging configuration
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load global settings
settings = Settings()
logger.setLevel(getattr(logging, settings.log_level.upper()))


@click.group()
@click.option("--log-level", default=settings.log_level, help="Set logging level")
@click.option("--force", is_flag=True, help="Force refresh of cached data")
@click.pass_context
def main(ctx, log_level, force):
    """
    ExoLife CLI
    """
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    ctx.obj["force"] = force

    # Update logging level
    logger.setLevel(getattr(logging, log_level.upper()))

    # Update global settings
    settings.log_level = log_level
    settings.force_refresh = force


def load_commands():
    """
    Auto-discover and register click commands from src/exolife/plugins/cli/*.py
    Each plugin module must define a top-level `cli` click.Command.
    """
    plugins_path = pathlib.Path(__file__).parent / "plugins" / "cli"
    package = "exolife.plugins.cli"
    for _, module_name, _ in pkgutil.iter_modules([str(plugins_path)]):
        full_name = f"{package}.{module_name}"
        try:
            module = importlib.import_module(full_name)
            cmd = getattr(module, "cli", None)
            if isinstance(cmd, click.Command):
                main.add_command(cmd)
        except Exception as e:
            logger.error(f"Failed to load plugin {full_name}: {e}")


# Load all plugin commands
load_commands()

if __name__ == "__main__":
    main()
