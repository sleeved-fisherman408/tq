import click

from tq.cli.models import list_cmd, pull, remove
from tq.cli.recommend import recommend_cmd
from tq.cli.start import start
from tq.cli.stop_status import status, stop
from tq.cli.system import system


@click.group()
@click.version_option(package_name="tq-serve")
def cli() -> None:
    """tq — Run local LLMs with maximum context on minimum hardware."""
    pass


cli.add_command(start)
cli.add_command(pull)
cli.add_command(list_cmd, name="list")
cli.add_command(remove)
cli.add_command(system)
cli.add_command(recommend_cmd, name="recommend")
cli.add_command(stop)
cli.add_command(status)
