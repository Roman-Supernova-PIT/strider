"""How ``strider classify`` looks: a live line while it runs, and each result as a ruled report. In a terminal it uses
colour, unless NO_COLOR is set; written to a file or a log, it is plain text."""

import logging
import os
import shutil
import sys
import threading
import time
import warnings

from .io import _tier
from .model import _duration

WIDTH = 79
RULE, TICK, CROSS, ARROW, SPINNER = "─", "✓", "✗", "→", "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
BOLD, DIM, RED, GREEN, YELLOW, CYAN = "1", "2", "31", "32", "33", "36"
STRENGTH = {"high": GREEN, "medium": YELLOW, "low": RED}
TIER = {"green": GREEN, "yellow": YELLOW, "orange": "38;5;166", "red": RED}  # the match's colours, and orange
GAP = "   "  # between items on a line


def interactive():
    """Whether ``strider`` is talking to a person: both output streams are a terminal."""
    return all(stream.isatty() for stream in (sys.stdout, sys.stderr)) and os.environ.get("TERM") != "dumb"


class Style:
    """Colour for one stream: only in a terminal, and not with NO_COLOR set."""

    def __init__(self, stream):
        self.colour = stream.isatty() and "NO_COLOR" not in os.environ

    def __call__(self, text, *codes):
        return f"\033[{';'.join(codes)}m{text}\033[0m" if self.colour and codes else text


def report(result, name, style, figure=None):
    """A result's report between two rules, indented, with its redshift range on a line of its own and, if given, the
    file of its figure. The top class is green, yellow, orange or red by its probability (`strider.io._tier`), and
    the match green, yellow or red."""
    label = f" {name} "
    label = label if len(label) <= WIDTH - 6 else label[: WIDTH - 8] + "… "
    lines = [style(RULE * 2, DIM) + style(label, BOLD) + style(RULE * (WIDTH - 2 - len(label)), DIM)]
    current = None  # the label a line without one continues
    for line in result.report(name).splitlines()[1:]:
        key, value = line[:10].strip(), line[10:]
        current = key or current
        if current == "Redshift":
            value, _, interval = value.partition(", ")
            lines += [_line(style, key, value), _line(style, "", interval)]
            continue
        if current == "Class":
            top, _, rest = value.partition(GAP)
            value = style(top, TIER[_tier(result._ranked()[0][1])], BOLD) + (GAP + rest if rest else "")
        elif current == "Match":
            value = ", ".join(_match_part(style, part) for part in value.split(", "))
        elif current == "Checks":
            value = style(value, YELLOW)
        lines.append(_line(style, key, value))
    lines.append(style(RULE * WIDTH, DIM))
    if figure is not None:
        lines.append(_line(style, "Figure", str(figure)))
    return "\n".join(lines)


def _line(style, key, value):
    return "   " + style(f"{key:<10}", DIM) + value


def _match_part(style, part):
    """The match strength in its colour; a warning in yellow."""
    return style(part, STRENGTH[part], BOLD) if part in STRENGTH else style(part, YELLOW)


def finished(style, done, failed, seconds, output=None, held=None):
    """The line that ends a run: how many this session classified and failed, how long it took, where to, and how
    many the file ``held`` in all when that is more (a resumed run)."""
    noun = "" if output is not None else " transient" if done == 1 else " transients"
    counts = f"{done}{noun} classified" + (f", {failed} failed," if failed else "")
    line = style(TICK, GREEN, BOLD) + f" {counts} in {_duration(seconds)}"
    if output is not None:
        line += f" {ARROW} {output}" + (f" ({held} in the file)" if held is not None and held > done + failed else "")
    return line


class LiveLine(logging.Handler):
    """A run's progress as one line on stderr that updates in place, from the ``strider`` log's records. Warnings,
    and with ``verbose`` each transient's line, print above it."""

    def __init__(self, verbose=False, name=None, workers=1, total=None):
        super().__init__(logging.DEBUG)
        self.verbose, self.name, self.workers, self.total = verbose, name, workers, total
        self.done = self.failed = 0
        self.style = Style(sys.stderr)
        self.started = self.last = time.monotonic()  # the start, and when the last transient finished
        self.drawing = threading.Lock()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        log = logging.getLogger("strider")
        self.saved = log.level, log.propagate, warnings.showwarning
        log.setLevel(logging.DEBUG)
        log.propagate = False  # the line replaces the log's progress lines
        log.addHandler(self)
        warnings.showwarning = self._warning
        self.thread.start()
        return self

    def __exit__(self, *exception):
        self.stopped.set()
        self.thread.join()
        with self.drawing:
            self._clear()
        log = logging.getLogger("strider")
        log.removeHandler(self)
        log.level, log.propagate, warnings.showwarning = self.saved

    def emit(self, record):
        if getattr(record, "done", self.done) != self.done:
            self.last = time.monotonic()
        self.done = getattr(record, "done", self.done)
        self.failed = getattr(record, "failed", self.failed)
        if record.levelno >= logging.WARNING:
            self._above(record.getMessage(), RED if record.levelno >= logging.ERROR else YELLOW)
        elif record.levelno == logging.DEBUG and self.verbose:
            self._above(record.getMessage())

    def text(self):
        """How far the run has got: how many are done (of how many, when known), failed, the time so far, the time
        each takes, and the time left. The time each takes is measured when a transient finishes, and the time left
        counts down from then."""
        now = time.monotonic()
        took = now - self.started
        total = self.total if self.total is not None and self.total >= self.done else None
        if not self.done:
            if self.name:
                return f"Classifying {self.name}{GAP}{_duration(took)}"
            what = f"{total} transient{'s' if total != 1 else ''}" if total is not None else "transients"
            workers = f"{self.workers} worker{'s' if self.workers > 1 else ''}"
            return f"Classifying {what} on {workers}{GAP}{_duration(took)}"
        each = (self.last - self.started) / self.done
        parts = [f"{self.done} of {total} done" if total is not None else f"{self.done} done"]
        parts += [f"{self.failed} failed"] if self.failed else []
        parts += [_duration(took), f"{_duration(each)} each"]
        if total is not None and total > self.done:
            left = each * (total - self.done) - (now - self.last)
            parts.append(f"{_duration(left)} left" if left >= 1.0 else "finishing")
        return GAP.join(parts)

    def _warning(self, message, category, filename, lineno, file=None, line=None):
        self._above(f"warning: {message}", YELLOW)

    def _above(self, text, *codes):
        with self.drawing:
            self._clear()
            sys.stderr.write(self.style(text, *codes) + "\n")
            self._draw()

    def _run(self):
        while not self.stopped.wait(0.1):
            with self.drawing:
                self._draw()

    def _draw(self):
        frame = SPINNER[int(10 * (time.monotonic() - self.started)) % len(SPINNER)]
        room = shutil.get_terminal_size().columns - 3  # never wrap, or the line cannot be redrawn in place
        sys.stderr.write("\r\033[K" + self.style(frame, CYAN) + " " + self.text()[:room])
        sys.stderr.flush()

    def _clear(self):
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
