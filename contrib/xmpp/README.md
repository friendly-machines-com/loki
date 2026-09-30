# Loki XMPP bridge

An optional, independently installable POSIX daemon connecting Loki's terminal
Unix-socket protocol v1 to one private XMPP group-chat room. It does not import
Loki, launch agent processes, manage Prosody, create accounts, or configure rooms.
Slixmpp is a dependency of this project only; Loki itself remains dependency-free.

## Install and configure

Requires Python 3.11 or later. From the repository root:

```sh
python3 -m venv ~/.local/share/loki-xmpp/venv
~/.local/share/loki-xmpp/venv/bin/pip install ./contrib/xmpp
mkdir -p ~/.config/loki-xmpp ~/.local/state/loki-xmpp
chmod 700 ~/.config/loki-xmpp ~/.local/state/loki-xmpp
cp contrib/xmpp/example.toml ~/.config/loki-xmpp/config.toml
chmod 600 ~/.config/loki-xmpp/config.toml
```

Edit the configuration: replace USER and UID in paths, set the actual room,
bot account, and allowed command-author JIDs. Create a separate password file,
mode `0600`, containing the bot password (an optional final newline is removed).
Use an editor rather than placing the password in shell arguments/history.
Create the bot account once with Prosody's normal administration if necessary;
the daemon never calls `prosodyctl`. Grant the bot ordinary room membership.
Do not make it a server administrator or allow it as a command author.

```sh
~/.local/share/loki-xmpp/venv/bin/loki-xmpp-bridge \
  --config ~/.config/loki-xmpp/config.toml --check-config
~/.local/share/loki-xmpp/venv/bin/loki-xmpp-bridge \
  --config ~/.config/loki-xmpp/config.toml
```

`--check-config` validates local configuration, password permissions, and TLS
trust configuration, without connecting or creating runtime/state directories.
It does not check credentials against Prosody or verify the room.

Then start any number of terminal sessions:

```sh
./loki.py --bridge-socket /run/user/UID/loki-xmpp/bridge.sock
```

The daemon creates private runtime/state directories and a mode `0600` socket.
Existing parent directories must be owned by its UID and mode `0700` (or more
restrictive). Run the bridge and Loki under the same UID for this initial setup.
Linux checks peer credentials as well; other POSIX platforms rely on filesystem
permissions. Singleton locks protect both the socket and state database. Active
listeners, symlinks, and unrelated files are never blindly removed.

For a user service, copy `loki-xmpp-bridge.service` to
`~/.config/systemd/user/`, adjust it if needed, and run:

```sh
systemctl --user daemon-reload
systemctl --user enable --now loki-xmpp-bridge
```

Configure user lingering separately if the daemon should outlive login sessions.
Stop with SIGINT/SIGTERM; outstanding durable state remains for recovery.

## Required Prosody room

The configured room must already exist and advertise non-anonymity,
members-only admission, persistence, MAM v2, and stable stanza IDs. The bot
refuses to operate if those features are missing, or if joining unexpectedly
creates a room or changes its requested nickname. It requests no join history
and never executes delayed/MAM/forwarded messages. Changes to room configuration
or removal of the bot disconnect it and force verification again.

Example defaults for *new* rooms:

```lua
Component "conference.ci.friendly-machines.com" "muc"
    modules_enabled = { "muc_mam" }
    restrict_room_creation = true
    muc_room_default_public = false
    muc_room_default_members_only = true
    muc_room_default_persistent = true
    muc_room_default_public_jids = true
    muc_room_default_history_length = 20
    max_history_messages = 20
    muc_log_by_default = true
    muc_log_expires_after = "never"
```

For an existing room, its owner must configure members-only, persistent,
non-anonymous (`muc#roomconfig_whois = anyone`), non-public/discoverable, and
archiving (`muc#roomconfig_enablearchiving = true`), and grant membership to
human accounts and the bot. Defaults do not overwrite existing room settings.
MAM and TLS certificate configuration depend on the installed Prosody version;
service discovery and real message IDs are checked at runtime, not assumed.
Enable `smacks` on the account host. No account-level MAM is required for this
room-only adapter. No plaintext or certificate-verification bypass is offered.
A private CA can be provided explicitly with `ca_file`; connecting to loopback
still verifies the bot JID's domain, not the loopback IP.

## Chat interface

Only live messages from an explicitly allowed **real bare JID** can issue
commands. Room nicknames alone never authorize anything. Membership does not
imply command authorization. The bot must have a current occupant/JID mapping;
missing identity, wrong room, bot echoes, and historical messages are ignored.
Messages also need exactly one stable stanza ID issued by the configured room.
A missing ID (including archive-storage failure) prevents execution.

```text
/sessions
/help
/to s1 Run the tests
/to s2 /models glm
/to s2 /providers "GLM-5.2"
/to s2 /model "GLM-5.2" --provider openrouter
/to s1 /bridge resume
```

Ordinary unaddressed conversation is ignored. There is no global selected
session, implicit last speaker, or broadcast prompt. `/to` strips only the
addressing prefix and forwards the remaining text to Loki, which decides which
slash commands/skills it supports. Each session has a durable alias (`s1`, `s2`,
...). Reconnection of the same live instance retains its alias. A new Loki
runtime receives a new alias even when resuming the same saved conversation.
Aliases are never automatically reused. Offline/unknown targets are rejected;
commands do not wait indefinitely for a replacement session.

Replies are tagged with the alias. Acceptance, running, completion/failure, and
remote pause are visible. Final responses and local command results are posted,
not every token/tool result. Keyboard prompt text is mirrored on turn completion;
remote prompt text is not echoed unnecessarily. Long output is split into
UTF-8-byte-bounded numbered messages. Invalid XML characters are replaced.

## Durability and failure semantics

SQLite holds aliases, incoming message/command identities, received Loki events,
and an ordered XMPP outbox. Local transactions are small and synchronous,
including durable commit before socket acknowledgements. This necessary local
filesystem operation is not offloaded to threads; all networking uses asyncio.
A database is bound to one bot/room and cannot silently be reused elsewhere.

* Room stanza IDs deduplicate incoming messages. Client origin IDs additionally
  deduplicate within an authenticated author's identity, never across authors.
  Conflicting ID reuse cannot execute a different command.
* A command is durably marked `submitted` **before** attempting its socket write.
  A crash/write failure after this point is uncertain, not permission to retry.
* Reconnection reconciles commands against Loki's retained input snapshot.
  Missing submitted/accepted/running IDs become uncertain and are not resubmitted.
  Durable, not-yet-submitted commands may execute only against their original
  live instance if it is still online; they never migrate to another runtime.
* Loki events are committed together with their generated outbox messages before
  cumulative acknowledgement. Event replay cannot duplicate generated output.
  Snapshots represent current state; older replay does not roll that state back.
* Explicit replay gaps are reported. Partial text is marked incomplete when
  chunk numbering reveals missing chunks. This cannot recover events which
  have fallen outside Loki's bounded journal.
* The XMPP outbox is sent in order, one message awaiting confirmation at a time.
  A live, authenticated self-echo with a room-issued archive ID confirms delivery;
  merely calling `send()` or receiving a stream acknowledgement is insufficient.
* XMPP disconnect/echo timeout preserves the outbox and reconnects with backoff.
  A crash after server delivery but before local confirmation can cause a
  duplicate post, using the same origin ID. This is **not exactly-once delivery**.
  Stream management is enabled when available, but recovery does not depend on
  maintaining Slixmpp's in-memory stream-resumption state across reconnects.
* Historical catch-up is never executable input. Messages sent while the bot is
  offline must be explicitly reissued as new commands, not replayed from MAM.

Defaults: 64 local connections; 128 pending bridge commands; 10,000 outbox
messages; 256 MiB SQLite database; 4 MiB assembled result; 3,000 UTF-8 bytes per
XMPP message; 30-second network/echo timeouts. Loki's frame/prompt limits remain
1 MiB/64 KiB. An incoming command queue overflow produces a durable rejection.
Outbox/database exhaustion or loss of durable storage stops the bridge without
acknowledging the failing event. Increase capacity/fix storage before restarting;
never delete the database merely to clear an error. SQLite rollback journals can
need additional temporary disk space beyond the database size limit.

Raw events and command IDs are retained; no automatic pruning erases
execution-deduplication evidence. Monitor disk usage and back up the state.
Results exceeding `max_result_bytes` produce an explicit notice; their events
remain in the database. Archive retention and storage backups remain separate
Prosody responsibilities. Neither MAM nor stream management makes local agent
execution crash-proof.

## Security boundary

The bridge owns only its bot credentials. Loki owns agent execution and model
credentials; none are exchanged through this protocol. The proxy never spawns
shell commands or interprets model output as input. Incoming XMPP identity is
captured synchronously before async routing, so later nickname reuse cannot
re-authorize an already received message.

State, passwords, prompts, and responses are sensitive. Keep them outside the
workspace and restrict room membership. **Same-UID agent tools may read these
files or connect to the socket**: Unix permissions are not a sandbox against
that UID. Use an outer VM/container/filesystem boundary if required. Restrict
Prosody client access to loopback/WireGuard, disable unnecessary federation/open
registration, and require TLS. Room admission, command allowlisting, and network
access are independent safeguards.

## Development

Install this project editable into its own environment, then run from its directory:

```sh
python3 -m pip install -e .
python3 -m unittest discover -s tests
python3 -m flake8 --max-line-length=110 src tests
python3 -m build
```

Tests use real local Unix sockets, actual Slixmpp stanza parsing, and fake network
adapters. They need neither a Prosody account nor live credentials. A live
room/Conversations check is a separate deployment verification.
