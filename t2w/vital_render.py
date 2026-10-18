"""Loading a .vital preset into the Vital VST3 with pedalboard and rendering a note.

pedalboard's raw_state is JUCE's VST3 state XML. The <IComponent> element holds JUCE-style base64 of a VST2-compatible
chunk (176-byte VstW/CcnK/FBCh header) followed by the preset JSON (the same format as a .vital file).
"""
import json

import numpy as np

_JUCE_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
_JUCE_IDX = {c: i for i, c in enumerate(_JUCE_B64)}
_CHUNK_HDR = 176


def _juce_b64_decode(s):
    size_str, data = s.split(".", 1)
    out = bytearray()
    for k in range(0, len(data), 4):
        n = 0
        for j, ch in enumerate(data[k:k + 4]):
            n |= _JUCE_IDX[ch] << (6 * j)
        out += n.to_bytes(3, "little")
    return bytes(out[:int(size_str)])


def _juce_b64_encode(b):
    n_chars = (len(b) * 8 + 5) // 6
    padded = b + b"\x00" * (-len(b) % 3)
    chars = []
    for k in range(0, len(padded), 3):
        n = int.from_bytes(padded[k:k + 3], "little")
        chars += [_JUCE_B64[(n >> (6 * j)) & 63] for j in range(4)]
    return f"{len(b)}." + "".join(chars[:n_chars])


def _split_state(raw):
    start = raw.index(b"<IComponent>") + len(b"<IComponent>")
    end = raw.index(b"</IComponent>")
    return raw[:start], _juce_b64_decode(raw[start:end].decode("ascii")), raw[end:]


def load_vital(vst3_path):
    import pedalboard
    return pedalboard.load_plugin(str(vst3_path))


def get_preset(plugin):
    _, chunk, _ = _split_state(plugin.raw_state)
    return json.JSONDecoder().raw_decode(chunk[_CHUNK_HDR:].decode("utf-8", "replace"))[0]


def set_preset(plugin, preset):
    head, chunk, tail = _split_state(plugin.raw_state)
    body = json.dumps(preset).encode("utf-8")
    hdr = bytearray(chunk[:_CHUNK_HDR])
    hdr[20:24] = (len(hdr) - 24 + len(body)).to_bytes(4, "big")
    hdr[172:176] = len(body).to_bytes(4, "big")
    xml = head[8:] + _juce_b64_encode(bytes(hdr) + body).encode("ascii") + tail.rstrip(b"\x00")
    plugin.raw_state = b"VC2!" + len(xml).to_bytes(4, "little") + xml + b"\x00"


def render_note(plugin, note=60, velocity=100, duration=2.0, sr=44100, release_margin=0.15):
    """Render one note (default C4, 2 s, note off at duration - 0.15 s). Returns (2, n) float32."""
    class _Midi:
        def __init__(self, status, n, v, t):
            self._b, self.time = bytes([status, n, v]), t

        def bytes(self):
            return self._b

    plugin.reset()
    msgs = [_Midi(0x90, note, velocity, 0.0), _Midi(0x80, note, 0, duration - release_margin)]
    return plugin.process(msgs, duration=duration, sample_rate=sr, num_channels=2).astype(np.float32)
