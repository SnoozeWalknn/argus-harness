"""A small GBNF recogniser, used by the mock server to check grammar-mode output.

Parses the llama.cpp GBNF dialect (literals, character classes with ranges
and negation, rule references, groups, ``| * + ?``, ``{m}``, ``{m,}``,
``{m,n}``, ``.``, comments) and decides whether a string matches ``root``.
Matching computes the set of end positions reachable from each (node,
position) pair with memoisation, and expands repetitions iteratively, so
ambiguous grammars and long inputs are fine.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


class GrammarError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class Lit:
    text: str


@dataclass(frozen=True, eq=False)
class Cls:
    ranges: tuple[tuple[int, int], ...]
    negated: bool

    def accepts(self, ch: str) -> bool:
        o = ord(ch)
        hit = any(lo <= o <= hi for lo, hi in self.ranges)
        return hit != self.negated


@dataclass(frozen=True, eq=False)
class Any_:
    pass


@dataclass(frozen=True, eq=False)
class Ref:
    name: str


@dataclass(frozen=True, eq=False)
class Rep:
    node: Node
    lo: int
    hi: int | None


@dataclass(frozen=True, eq=False)
class Seq:
    items: tuple[Node, ...]


@dataclass(frozen=True, eq=False)
class Alt:
    options: tuple[Node, ...]


Node = Lit | Cls | Any_ | Ref | Rep | Seq | Alt

_ESC = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    '"': '"',
    "[": "[",
    "]": "]",
    "-": "-",
    "^": "^",
    "/": "/",
}


class _Parser:
    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def error(self, msg: str) -> GrammarError:
        line = self.s.count("\n", 0, self.i) + 1
        return GrammarError(f"{msg} at line {line}: {self.s[self.i : self.i + 30]!r}")

    def ws(self, newlines: bool) -> None:
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "#":
                while self.i < len(self.s) and self.s[self.i] != "\n":
                    self.i += 1
            elif c in " \t\r" or (newlines and c == "\n"):
                self.i += 1
            else:
                break

    def name(self) -> str:
        j = self.i
        while self.i < len(self.s) and (self.s[self.i].isalnum() or self.s[self.i] in "-_"):
            self.i += 1
        if j == self.i:
            raise self.error("expected a rule name")
        return self.s[j : self.i]

    def rules(self) -> dict[str, Node]:
        out: dict[str, Node] = {}
        self.ws(True)
        while self.i < len(self.s):
            name = self.name()
            self.ws(False)
            if not self.s.startswith("::=", self.i):
                raise self.error("expected ::=")
            self.i += 3
            self.ws(True)
            out[name] = self.alt(top=True)
            self.ws(True)
        if "root" not in out:
            raise GrammarError("grammar has no root rule")
        for name in out:
            self._check_refs(out[name], out)
        return out

    def _check_refs(self, node: Node, rules: dict[str, Node]) -> None:
        if isinstance(node, Ref) and node.name not in rules:
            raise GrammarError(f"undefined rule {node.name!r}")
        for child in getattr(node, "items", ()) or getattr(node, "options", ()):
            self._check_refs(child, rules)
        if isinstance(node, Rep):
            self._check_refs(node.node, rules)

    def _at_rule_start(self) -> bool:
        """True if the text at i (after a newline) begins a new ``name ::=`` rule."""
        j = self.i
        while j < len(self.s) and (self.s[j].isalnum() or self.s[j] in "-_"):
            j += 1
        k = j
        while k < len(self.s) and self.s[k] in " \t":
            k += 1
        return j > self.i and self.s.startswith("::=", k)

    def alt(self, top: bool = False) -> Node:
        options = [self.seq(top)]
        while True:
            self.ws(not top)
            if top and self.i < len(self.s) and self.s[self.i] == "\n":
                # a rule body may continue on the next line if it starts with |
                j = self.i
                self.ws(True)
                if self.i < len(self.s) and self.s[self.i] == "|":
                    pass
                else:
                    self.i = j
                    break
            if self.i < len(self.s) and self.s[self.i] == "|":
                self.i += 1
                self.ws(True)
                options.append(self.seq(top))
            else:
                break
        return options[0] if len(options) == 1 else Alt(tuple(options))

    def seq(self, top: bool) -> Node:
        items: list[Node] = []
        while True:
            self.ws(not top)
            if self.i >= len(self.s):
                break
            c = self.s[self.i]
            if c in "|)" or c == "\n":
                break
            if top and self._at_rule_start() and items:
                break
            items.append(self.postfix(self.atom()))
        return items[0] if len(items) == 1 else Seq(tuple(items))

    def atom(self) -> Node:
        c = self.s[self.i]
        if c == '"':
            return Lit(self.literal())
        if c == "[":
            return self.charclass()
        if c == "(":
            self.i += 1
            self.ws(True)
            node = self.alt()
            self.ws(True)
            if self.i >= len(self.s) or self.s[self.i] != ")":
                raise self.error("expected )")
            self.i += 1
            return node
        if c == ".":
            self.i += 1
            return Any_()
        if c.isalnum() or c in "-_":
            return Ref(self.name())
        raise self.error(f"unexpected {c!r}")

    def postfix(self, node: Node) -> Node:
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "*":
                node, self.i = Rep(node, 0, None), self.i + 1
            elif c == "+":
                node, self.i = Rep(node, 1, None), self.i + 1
            elif c == "?":
                node, self.i = Rep(node, 0, 1), self.i + 1
            elif c == "{":
                j = self.s.index("}", self.i)
                spec = self.s[self.i + 1 : j]
                if "," in spec:
                    a, b = spec.split(",", 1)
                    node = Rep(node, int(a or 0), int(b) if b.strip() else None)
                else:
                    node = Rep(node, int(spec), int(spec))
                self.i = j + 1
            else:
                break
        return node

    def _char(self) -> str:
        c = self.s[self.i]
        if c != "\\":
            self.i += 1
            return c
        e = self.s[self.i + 1]
        if e == "x":
            v = chr(int(self.s[self.i + 2 : self.i + 4], 16))
            self.i += 4
            return v
        if e == "u":
            v = chr(int(self.s[self.i + 2 : self.i + 6], 16))
            self.i += 6
            return v
        if e == "U":
            v = chr(int(self.s[self.i + 2 : self.i + 10], 16))
            self.i += 10
            return v
        self.i += 2
        if e not in _ESC:
            raise self.error(f"unknown escape \\{e}")
        return _ESC[e]

    def literal(self) -> str:
        self.i += 1
        out = []
        while True:
            if self.i >= len(self.s):
                raise self.error("unterminated string")
            if self.s[self.i] == '"':
                self.i += 1
                return "".join(out)
            out.append(self._char())

    def charclass(self) -> Cls:
        self.i += 1
        negated = self.s[self.i] == "^"
        if negated:
            self.i += 1
        ranges = []
        while True:
            if self.i >= len(self.s):
                raise self.error("unterminated character class")
            if self.s[self.i] == "]":
                self.i += 1
                return Cls(tuple(ranges), negated)
            lo = self._char()
            if self.s[self.i] == "-" and self.s[self.i + 1] != "]":
                self.i += 1
                hi = self._char()
                ranges.append((ord(lo), ord(hi)))
            else:
                ranges.append((ord(lo), ord(lo)))


class Grammar:
    def __init__(self, text: str):
        self.rules = _Parser(text).rules()

    def matches(self, text: str, rule: str = "root") -> bool:
        m = _Matcher(self.rules, text)
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(limit, 20000))
        try:
            return len(text) in m.match(Ref(rule), 0)
        finally:
            sys.setrecursionlimit(limit)


class _Matcher:
    def __init__(self, rules: dict[str, Node], text: str):
        self.rules = rules
        self.s = text
        self.memo: dict[tuple[int, int], frozenset[int]] = {}
        self.active: set[tuple[int, int]] = set()

    def match(self, node: Node, pos: int) -> frozenset[int]:
        key = (id(node), pos)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        if key in self.active:  # left recursion: treat as no match
            return frozenset()
        self.active.add(key)
        try:
            out = frozenset(self._match(node, pos))
        finally:
            self.active.discard(key)
        self.memo[key] = out
        return out

    def _match(self, node: Node, pos: int) -> set[int]:
        s = self.s
        if isinstance(node, Lit):
            return {pos + len(node.text)} if s.startswith(node.text, pos) else set()
        if isinstance(node, Cls):
            return {pos + 1} if pos < len(s) and node.accepts(s[pos]) else set()
        if isinstance(node, Any_):
            return {pos + 1} if pos < len(s) else set()
        if isinstance(node, Ref):
            return set(self.match(self.rules[node.name], pos))
        if isinstance(node, Alt):
            out: set[int] = set()
            for o in node.options:
                out |= self.match(o, pos)
            return out
        if isinstance(node, Seq):
            frontier = {pos}
            for item in node.items:
                nxt: set[int] = set()
                for p in frontier:
                    nxt |= self.match(item, p)
                if not nxt:
                    return set()
                frontier = nxt
            return frontier
        if isinstance(node, Rep):
            return self._rep(node, pos)
        raise TypeError(node)

    def _rep(self, node: Rep, pos: int) -> set[int]:
        results: set[int] = set()
        frontier = {pos}
        seen = {pos}
        count = 0
        while True:
            if count >= node.lo:
                results |= frontier
            if node.hi is not None and count >= node.hi:
                break
            nxt: set[int] = set()
            for p in frontier:
                for e in self.match(node.node, p):
                    if e > p or count < node.lo:
                        nxt.add(e)
            count += 1
            if count > node.lo:
                nxt -= seen
                seen |= nxt
            if not nxt:
                break
            frontier = nxt
        return results
