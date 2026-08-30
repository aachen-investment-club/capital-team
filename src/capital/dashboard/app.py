"""
Dash app factory. gunicorn target: capital.dashboard.app:server

Pages live in capital/dashboard/pages/, one file per page, auto-registered
via dash.register_page (see pages/_template.py for the pattern).
"""
import diskcache
from dash import Dash, DiskcacheManager

import capital.theme  # noqa: F401, registers the "capital" plotly template
from capital.settings import settings
from capital.dashboard.shell import build_shell
from capital.jobs import queue as jobs

# Background-callback manager for long computations (optimiser, GARCH fits).
settings.cache_dir.mkdir(parents=True, exist_ok=True)
_background = DiskcacheManager(diskcache.Cache(str(settings.cache_dir / "background")))


def create_app() -> Dash:
    app = Dash(
        __name__,
        use_pages=True,
        title="Capital Team Dashboard",
        update_title=None,
        background_callback_manager=_background,
        suppress_callback_exceptions=True,
    )
    app._favicon = "logo-icon.png"
    app.layout = build_shell()
    # Background job queue (factor-model runs). Work happens in subprocesses, so
    # this thread only starts and reaps them; it never competes with a request.
    jobs.start_pump()
    return app


app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8050, use_reloader=False)
