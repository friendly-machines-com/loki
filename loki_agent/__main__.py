from .credentials import capture_process_credentials


def main() -> int:
    # Scrub the initial exec environment before importing the terminal
    # frontend and the rest of Loki.
    capture_process_credentials()
    from .terminal_frontend import main as terminal_main
    return terminal_main()


if __name__ == "__main__":
    raise SystemExit(main())
