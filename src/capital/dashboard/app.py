"""
Dash app factory. gunicorn target: capital.dashboard.app:server

Pages live in capital/dashboard/pages/ — one file per page, auto-registered
via dash.register_page (see pages/_template.py for the pattern).
"""
import diskcache
from dash import Dash, DiskcacheManager

import capital.theme  # noqa: F401 — registers the "capital" plotly template
from capital.settings import settings
from capital.dashboard.shell import build_shell

# Background-callback manager for long computations (optimiser, GARCH fits).
settings.cache_dir.mkdir(parents=True, exist_ok=True)
_background = DiskcacheManager(diskcache.Cache(str(settings.cache_dir / "background")))


def create_app() -> Dash:
    app = Dash(
        __name__,
        use_pages=True,
        title="AIC Capital Dashboard",
        update_title=None,
        background_callback_manager=_background,
        suppress_callback_exceptions=True,
    )
    app.layout = build_shell()
    return app


app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8050)
