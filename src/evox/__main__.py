"""Allow ``python -m evox`` to use the same CLI as the installed command."""
from evox.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
