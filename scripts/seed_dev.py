"""Seed a development API key and a demo experiment.

Run with ``make seed``. The app factory already bootstraps the documented
development key on first start; this script is for creating a named key and a
stable experiment for manual testing.
"""

from __future__ import annotations

from sqlalchemy import select
from traintrack.db import get_session, init_db
from traintrack.models.experiment import Experiment
from traintrack.services.auth import create_api_key


def main() -> None:
    """Create a named API key and demo experiment if they do not exist."""
    init_db()
    session = get_session()
    try:
        record, raw = create_api_key(session, "manual-dev")
        experiment = session.execute(
            select(Experiment).where(Experiment.name == "mnist-baseline")
        ).scalar_one_or_none()
        if experiment is None:
            experiment = Experiment(
                name="mnist-baseline",
                description="MNIST CNN learning-rate and batch-size sweep",
            )
            session.add(experiment)
            session.flush()
        session.commit()
        print(f"API key ({record.key_prefix}...): {raw}")
        print(f"Experiment id: {experiment.id}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
