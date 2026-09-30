"""Standalone bridge executable; never launches or imports Loki."""

import argparse
import asyncio
import logging
import os
import signal
import ssl
import sqlite3
import sys

from .configs import Config, ConfigurationError


async def serve(config, password):
    from .routes import Router
    from .sockets import SocketServer
    from .stores import Store
    from . import xmpps

    store = Store(config)
    router = Router(config, store)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    handlers = {}
    tasks = []
    try:
        for name in (signal.SIGTERM, signal.SIGINT):
            handlers[name] = signal.getsignal(name)
            loop.add_signal_handler(name, stop.set)
        async with SocketServer(config, router):
            tasks = [asyncio.create_task(router.worker()),
                     asyncio.create_task(xmpps.run(config, password, router)),
                     asyncio.create_task(stop.wait()),
                     asyncio.create_task(router.failed.wait())]
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if router.failure is not None:
                raise RuntimeError('Bridge durability/processing failed; check storage capacity and logs')
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for name, previous in handlers.items():
            loop.remove_signal_handler(name)
            signal.signal(name, previous)
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Loki Unix-socket to private XMPP room bridge')
    parser.add_argument('--config', required=True, help='explicit TOML configuration file')
    parser.add_argument('--check-config', action='store_true', help='validate local files without networking')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    if os.name != 'posix':
        print('loki-xmpp-bridge: POSIX Unix sockets are required', file=sys.stderr)
        return 2
    try:
        config = Config.load(args.config)
        password = config.password()
        ssl.create_default_context(cafile=config.ca_file)
        if args.check_config:
            print('Configuration and password file valid; network/room not checked.')
            return 0
        asyncio.run(serve(config, password))
    except (OSError, ValueError, ConfigurationError, RuntimeError, sqlite3.Error) as error:
        print(f'loki-xmpp-bridge: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
