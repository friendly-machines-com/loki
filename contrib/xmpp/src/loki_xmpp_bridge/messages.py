"""Plain-text room output with explicit session tags and UTF-8 byte limits."""


def xml_text(text):
    # XML normalizes CR/CRLF on receipt; persist the same canonical body so
    # exact self-echo confirmation cannot stall on Windows-style line endings.
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return ''.join(char if (char in '\t\n\r' or 0x20 <= ord(char) <= 0xd7ff
                            or 0xe000 <= ord(char) <= 0xfffd
                            or 0x10000 <= ord(char) <= 0x10ffff)
                   else '\ufffd' for char in text)


def split_text(prefix, text, maximum):
    prefix, text = xml_text(prefix), xml_text(text)
    budget = maximum - len(prefix.encode('utf-8')) - 32
    if budget < 4:
        raise ValueError('Message prefix exceeds configured size')
    parts = []
    current = []
    size = 0
    for char in text:
        width = len(char.encode('utf-8'))
        if size + width > budget:
            parts.append(''.join(current))
            current, size = [], 0
        current.append(char)
        size += width
    if current or not parts:
        parts.append(''.join(current))
    if len(parts) == 1:
        return [prefix + parts[0]]
    return [f'{prefix}({index}/{len(parts)})\n{part}'
            for index, part in enumerate(parts, 1)]


def render(alias, event, store):
    kind = event['type']
    limit = store.config.message_bytes
    if kind == 'prompt_accepted' and not event['duplicate']:
        return [f"[{alias}] Queued (request {event['input_id'][:8]})."]
    if kind == 'prompt_rejected':
        return [f"[{alias}] Rejected: {event['reason']}."]
    if kind == 'turn_started':
        model = event.get('model') or 'unknown model'
        provider = event.get('provider') or 'configured connection'
        return split_text(f'[{alias}] Running: ', f'{model} via {provider}.', limit)
    if kind in ('turn_finished', 'command_finished'):
        result = []
        if kind == 'turn_finished' and event['origin'] == 'keyboard':
            text, incomplete = store.chunks(event['instance_id'], event, 'input_chunk')
            if text:
                prefix = f'[{alias}] Keyboard prompt' + (' (incomplete)' if incomplete else '') + ':\n'
                result.extend(split_text(prefix, text, limit))
        text, incomplete = store.chunks(event['instance_id'], event, 'output_chunk')
        prefix = f"[{alias}] {event['outcome']}"
        if incomplete:
            prefix += ' (output incomplete)'
        if kind == 'turn_finished' and event['paused']:
            prefix += '; remote execution paused: /to ' + alias + ' /bridge resume'
        prefix += ':\n' if text else '.'
        result.extend(split_text(prefix, text, limit))
        return result
    if kind == 'session_state':
        return [f"[{alias}] Remote execution {'paused' if event['paused'] else 'resumed'}."]
    if kind == 'session_closing':
        return [f'[{alias}] Session closed.']
    if kind == 'protocol_error':
        return [f"[{alias}] Loki protocol error: {event['reason']}."]
    return []
