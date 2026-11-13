"""Run the TrainTrack API with the Flask development server."""

from __future__ import annotations

from traintrack.api.app import create_app
from traintrack.config import settings


def main() -> None:
    """Entry point for ``python -m traintrack.api``."""
    app = create_app()
    app.run(host=settings().api_host, port=settings().api_port, threaded=True)


if __name__ == "__main__":
    main()
