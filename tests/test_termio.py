"""Tests for the terminal front end: the keys, and how they are spelt.

The decoder is the half of `termio` that needs no terminal at all — feed it
characters, read back symbolic `Key`s — so its pins live next to it rather than
among the status-line tests, which drive the same events through a StatusApp.
What the OUT half writes is still asserted over there, where a pinned region and
a run to pin it under are already staged.
"""

from llm_loop import termio as tio


def test_escape_decoder_names_the_special_keys_on_both_platforms():
    posix = tio._EscapeDecoder(scan_codes=False)

    assert [e.char for e in posix.feed("s")] == ["s"]
    assert posix.feed("\x1b") == [] and posix.feed("[") == []
    assert [e.char for e in posix.feed("A")] == ["up"]
    assert posix.feed("\x1b") == []
    assert [e.char for e in posix.flush()] == ["\x1b"]     # a bare Esc

    windows = tio._EscapeDecoder(scan_codes=True)
    assert windows.feed("\xe0") == []
    assert [e.char for e in windows.feed("K")] == ["left"]


def test_the_word_wise_arrows_are_named_on_both_platforms():
    """Ctrl+Left/Right: two unrelated spellings, one symbolic key, so the line
    editor never learns which platform it is on."""
    posix = tio._EscapeDecoder(scan_codes=False)
    for char in "\x1b[1;5":
        assert posix.feed(char) == []
    assert [e.char for e in posix.feed("D")] == ["wordleft"]

    windows = tio._EscapeDecoder(scan_codes=True)
    assert windows.feed("\xe0") == []
    assert [e.char for e in windows.feed("t")] == ["wordright"]
    # …and the same letter without the lead byte is still just a letter.
    assert [e.char for e in windows.feed("t")] == ["t"]


def test_the_decoder_rejoins_a_utf16_surrogate_pair():
    """msvcrt hands over code UNITS: an emoji in a pasted line arrives as two
    lone surrogates, each unprintable on its own and therefore dropped."""
    windows = tio._EscapeDecoder(scan_codes=True)

    assert windows.feed("\ud83d") == []
    assert [e.char for e in windows.feed("\ude80")] == ["🚀"]

    # A half pair that never completes is not a keypress, and must not glue
    # itself to whatever is typed next.
    windows.feed("\ud83d")
    assert windows.flush() == []
    assert [e.char for e in windows.feed("a")] == ["a"]


def test_escape_decoder_drops_sequences_nothing_understands():
    decoder = tio._EscapeDecoder(scan_codes=False)

    for char in "\x1b[200~":                 # bracketed paste: not a keypress
        events = decoder.feed(char)

    assert events == []
