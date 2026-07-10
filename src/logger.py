import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure standard console logging for the whole app."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


def get_logger(name: str = "chat_with_pdf_rag") -> logging.Logger:
    """Return a named logger after the shared logging config is installed."""
    setup_logging()
    return logging.getLogger(name)
