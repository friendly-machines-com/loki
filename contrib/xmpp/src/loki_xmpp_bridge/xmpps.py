"""Verified-TLS MUC adapter; account identity is captured before async routing."""

import asyncio
import logging
import random
import ssl
from xml.etree import ElementTree as ET

from slixmpp import ClientXMPP, JID

from .routes import STORAGE_ERRORS


logger = logging.getLogger(__name__)
SID = 'urn:xmpp:sid:0'
REQUIRED_FEATURES = {'muc_nonanonymous', 'muc_membersonly', 'muc_persistent',
                     'urn:xmpp:mam:2', SID}


def stanza_ids(message, room):
    trusted = [element.get('id') for element in message.xml.findall(f'{{{SID}}}stanza-id')
               if element.get('by') == room]
    if len(trusted) != 1 or not trusted[0] or len(trusted[0]) > 512:
        return None, None
    origins = message.xml.findall(f'{{{SID}}}origin-id')
    origin = origins[0].get('id') if len(origins) == 1 else None
    if origin is not None and (not origin or len(origin) > 512):
        origin = None
    return trusted[0], origin


def historical(message):
    # Reject archive wrappers as well as join history, even if a third party
    # forwards a historical stanza in a new message. They are never commands.
    for element in message.xml.iter():
        if element.tag in ('{urn:xmpp:delay}delay', '{jabber:x:delay}x',
                           '{urn:xmpp:forward:0}forwarded'):
            return True
        if element.tag.startswith('{urn:xmpp:mam:'):
            return True
    return False


class RoomClient(ClientXMPP):
    def __init__(self, config, password, router):
        super().__init__(config.jid + '/loki-xmpp-bridge', password)
        self.config = config
        self.router = router
        self.ready = asyncio.Event()
        self.stopped = asyncio.Event()
        self.join_task = None
        self.filter_task = None
        self.echoes = {}
        self.enable_direct_tls = False
        self.enable_starttls = True
        self.enable_plaintext = False
        self.ssl_context = ssl.create_default_context(cafile=config.ca_file)
        self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        for plugin in ('xep_0030', 'xep_0045', 'xep_0198', 'xep_0359', 'xep_0199'):
            self.register_plugin(plugin)
        self.add_event_handler('session_start', self.session_started)
        self.add_event_handler('disconnected', self.connection_stopped)
        self.add_event_handler('failed_auth', self.authentication_failed)
        self.add_event_handler('groupchat_message', self.receive_message)
        self.add_event_handler('groupchat_config_status', self.room_changed)
        self.add_event_handler(f'muc::{config.room}::self-presence', self.self_presence)

    async def _handle_stream_features(self, features):
        # Do not attempt even SCRAM authentication on an unencrypted stream.
        # STARTTLS, if offered, runs before SASL in Slixmpp's feature ordering.
        encrypted = self.transport is not None and self.transport.get_extra_info('ssl_object') is not None
        offered = features['features']
        if not encrypted and 'starttls' not in offered:
            logger.error('XMPP server did not offer required TLS; refusing authentication.')
            self.abort()
            self.stopped.set()
            return True
        return await super()._handle_stream_features(features)

    async def run_filters(self):
        # Slixmpp's output worker outlives disconnect; retain its task explicitly
        # so replacing a client on reconnect cannot leak it or queued stanzas.
        self.filter_task = asyncio.current_task()
        try:
            await super().run_filters()
        finally:
            self.filter_task = None

    def session_started(self, _event):
        if not self.stopped.is_set() and self.join_task is None:
            self.join_task = asyncio.create_task(self.join_room())

    async def room_features(self):
        info = await self.plugin['xep_0030'].get_info(
            jid=self.config.room, timeout=self.config.network_timeout)
        features = set(info['disco_info']['features'])
        if not REQUIRED_FEATURES <= features:
            missing = ', '.join(sorted(REQUIRED_FEATURES - features))
            raise ValueError('Room lacks required security/archive features: ' + missing)

    async def join_room(self):
        try:
            # Verify existence before joining: the bot must not create/configure
            # a room implicitly. Membership and room configuration are operator-owned.
            await self.room_features()
            presence, _subject, _occupants, _history = await self.plugin['xep_0045'].join_muc_wait(
                JID(self.config.room), self.config.nick, maxstanzas=0,
                timeout=self.config.network_timeout)
            if 201 in presence['muc']['status_codes'] or presence['from'].resource != self.config.nick:
                raise ValueError('Unexpected room creation or nickname assignment')
            await self.room_features()
            if self.stopped.is_set():
                raise ConnectionError('Disconnected while verifying room')
            self.ready.set()
            logger.info('XMPP room ready.')
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning('Room join/verification failed (%s).', type(error).__name__)
            self.stopped.set()
            self.abort()

    def connection_stopped(self, _event):
        self.ready.clear()
        self.stopped.set()
        for future in self.echoes.values():
            if not future.done():
                future.set_exception(ConnectionError('XMPP disconnected'))

    def authentication_failed(self, _event):
        logger.error('XMPP authentication failed; check the bot account/password.')
        self.stopped.set()
        self.abort()

    def self_presence(self, presence):
        if presence['type'] in ('unavailable', 'error'):
            self.connection_stopped(None)
            self.abort()

    def room_changed(self, message):
        if message['from'].bare == self.config.room and not message['from'].resource:
            # Only the room service, not an occupant's forged MUC extension,
            # can invalidate verification. Revalidate before further traffic.
            self.connection_stopped(None)
            self.abort()

    def receive_message(self, message):
        if (not self.ready.is_set() or message['type'] != 'groupchat'
                or message['from'].bare != self.config.room or historical(message)):
            return
        nick = message['from'].resource
        if not nick:
            return
        real = self.plugin['xep_0045'].get_jid_property(JID(self.config.room), nick, 'jid')
        role = self.plugin['xep_0045'].get_jid_property(JID(self.config.room), nick, 'role')
        if not real or role not in ('participant', 'moderator'):
            return
        author = JID(real).bare
        message_id, origin = stanza_ids(message, self.config.room)
        body = message['body']
        if message_id is None:
            # Advertising SID/MAM alone is insufficient: a missing archive ID
            # may indicate storage failure. Never execute that message.
            logger.warning('Ignoring room message without a trusted stable ID.')
            return
        try:
            if author == self.config.jid:
                if nick == self.config.nick and origin is not None:
                    delivered = self.router.store.delivered(origin, body)
                    if delivered:
                        future = self.echoes.get(origin)
                        if future is not None and not future.done():
                            future.set_result(None)
                        self.router.outgoing.set()
                return
            if author not in self.config.allowed_jids or nick == self.config.nick:
                return
            self.router.receive_chat(author, message_id, origin, body)
        except Exception as error:
            self.router.fail(error)

    async def send_confirmed(self, row):
        future = asyncio.get_running_loop().create_future()
        self.echoes[row['origin_id']] = future
        try:
            stanza = self.make_message(mto=self.config.room, mbody=row['body'], mtype='groupchat')
            stanza['id'] = row['origin_id']
            element = ET.SubElement(stanza.xml, f'{{{SID}}}origin-id')
            element.set('id', row['origin_id'])
            stanza.send()
            # A trusted self-echo with room archive ID confirms delivery to the
            # room, stronger than merely enqueueing or acknowledging a stream.
            await asyncio.wait_for(future, self.config.echo_timeout)
        finally:
            self.echoes.pop(row['origin_id'], None)
            if not future.done():
                future.cancel()

    async def pump_outbox(self):
        while True:
            self.router.outgoing.clear()
            row = self.router.store.next_outgoing()
            if row is None:
                await self.router.outgoing.wait()
                continue
            await self.send_confirmed(row)

    async def close(self):
        self.stopped.set()
        self.ready.clear()
        self.cancel_connection_attempt()
        if self.join_task is not None:
            self.join_task.cancel()
            await asyncio.gather(self.join_task, return_exceptions=True)
        await self.disconnect(wait=1, ignore_send_queue=True)
        if self.filter_task is not None:
            task = self.filter_task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for future in self.echoes.values():
            if not future.done():
                future.cancel()
        self.echoes.clear()


async def run(config, password, router, *, client_factory=RoomClient):
    delay = 1
    while True:
        client = client_factory(config, password, router)
        tasks = []
        try:
            await asyncio.wait_for(client.connect(config.host, config.port), config.network_timeout)
            tasks = [asyncio.create_task(client.ready.wait()),
                     asyncio.create_task(client.stopped.wait())]
            done, _pending = await asyncio.wait(
                tasks, timeout=config.network_timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done or client.stopped.is_set() or not client.ready.is_set():
                raise ConnectionError('XMPP session/room not ready')
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            tasks = [asyncio.create_task(client.pump_outbox()),
                     asyncio.create_task(client.stopped.wait())]
            delay = 1
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except asyncio.CancelledError:
            raise
        except STORAGE_ERRORS as error:
            router.fail(error)
            return
        except Exception as error:
            logger.warning('XMPP unavailable (%s); retaining outbox and retrying.', type(error).__name__)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.close()
        await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        delay = min(30, delay * 2)
