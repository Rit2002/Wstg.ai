import os
from contextlib import contextmanager, nullcontext

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()


# all returns True if both the env are present
LANGFUSE_ENABLED = all(
    os.getenv(key)
    for key in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    )
)

langfuse = get_client() if LANGFUSE_ENABLED else None


# The '*' forces the arguments after it to be keyword-only.
@contextmanager
def observe(
    name: str,
    *,
    as_type: str = "span",
    input_data: dict | None = None,
    metadata: dict | None = None,
    model: str | None = None
):
    """
    Create a Langfuse observation when configured.
    Otherwise, execute without tracing.
    """

    if not LANGFUSE_ENABLED or langfuse is None:
        with nullcontext() as observation:
            yield observation
        return

    kwargs = {
        "name" : name,
        "as_type": as_type
    }

    if input_data is not None:
        kwargs["input"] = input_data

    if metadata is not None:
        kwargs["metadata"] = metadata

    if model is not None:
        kwargs["model"] = model

    with langfuse.start_as_current_observation(**kwargs) as observation:
        try:
            yield observation
        except Exception as exc:
            observation.update(
                level="ERROR",
                status_message=f"{type(exc).__name__}: {exc}",
            )
            raise


def flush_traces():
    """Flush pending observations to Langfuse."""

    if LANGFUSE_ENABLED and langfuse is not None:
        langfuse.flush()