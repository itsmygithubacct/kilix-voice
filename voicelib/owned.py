"""Opt-in owned Avatar turns layered on the daemon's normal job lifecycle.

An owner is a client instance token. Owned speech never replaces other audio:
admission is refused as busy while anything is speaking or listening. Each
utterance token names exactly one request; a retry returns the same result and
never replays audio, and reusing the token for different speech is malformed.

The turn a client holds is `speak-<daemon instance>-<n>`, so a stop kept
across a daemon restart cannot match a new instance's speech. Owned states are
the job outcomes, with two refinements a client can act on: `superseded` when
a legacy `speak` replaced the turn and `expired` when the owner stopped
renewing its lease.
"""
import hashlib
import secrets
import time
from collections import OrderedDict

from . import jobs, protocol

OWNED_RESULTS_KEPT = 128
SPEECH_LEASE_S = 5.0
DICTATION_LEASE_S = 3.0

# Why a cancelled owned turn ended, when the owner did not stop it itself.
END_SUPERSEDED = "superseded"
END_EXPIRED = "expired"


def _signature(request):
    return hashlib.sha256(protocol.encode(
        {key: request[key] for key in ("text", "model", "voice", "rate")
         if key in request})).hexdigest()


class OwnedTurns:
    """Mixed into the daemon; expects its lock, turns, arbiter and player."""

    def _owned_state(self):
        # Created on first use so the daemon's constructor stays unchanged.
        if not hasattr(self, "_owned_entries"):
            self._owned_entries = OrderedDict()
            self._owned_instance = secrets.token_hex(8)
        return self._owned_entries

    def _owned_token(self, turn_id):
        """The wire turn for a daemon turn ID `speak-<n>`."""
        return f"speak-{self._owned_instance}-{turn_id.rpartition('-')[2]}"

    def _owned_payload(self, entry):
        turn = entry["turn"]
        if turn is None:        # nothing speakable: completed on admission
            return dict(turn=entry["token"], state="completed", playing=False, detail="")
        outcome = turn.outcome
        if outcome is None:
            player = getattr(self, "_player", None)
            playing = bool(self._speech is turn and player is not None and player.playing)
            state, detail = ("speaking" if playing else "preparing"), ""
        else:
            playing, detail = False, outcome.message
            state = outcome.outcome
            if state == jobs.OUTCOME_DEADLINE:
                state = END_EXPIRED
            elif state == jobs.OUTCOME_CANCELLED and getattr(turn, "owned_end", ""):
                state = turn.owned_end
        return dict(turn=entry["token"], state=state, playing=playing, detail=detail)

    def _owned_supersede(self):
        """Called under the lock just before a legacy read replaces speech."""
        turn = self._speech
        if turn is not None and getattr(turn, "owner", ""):
            turn.owned_end = END_SUPERSEDED

    def _owned_expire(self):
        with self._lock:
            now = time.monotonic()
            speech, dictation = self._speech, self._dictation
            if (speech is not None and getattr(speech, "owner", "")
                    and now >= speech.lease_until):
                speech.owned_end = END_EXPIRED
                self._cancel_speech()
            if (dictation is not None and getattr(dictation, "owner", "")
                    and now >= dictation.lease_until):
                dictation.abort = True
                dictation.stop.set()

    def _op_owned(self, request):
        op = request["op"]
        if op == "owned-speak":
            return self._owned_speak(request)
        if op == "owned-status":
            return self._owned_status(request)
        if op == "owned-stop":
            return self._owned_stop(request)
        if op == "owned-dictate":
            with self._lock:
                if self._arbiter.speaking:
                    return protocol.reply_error("Audio is busy", protocol.ERR_BUSY)
            return self._op_dictate(request)
        with self._lock:
            turn = self._dictation
            active = bool(turn and getattr(turn, "owner", "") == request["owner"])
            if active:
                if op == "owned-dictation-status":
                    turn.lease_until = time.monotonic() + DICTATION_LEASE_S
                else:
                    with turn.capture_lock:
                        turn.owned_finish.set()
                        if turn.capture is not None:
                            turn.capture.request_stop()
            return protocol.reply_ok(request["id"], **(
                {"active": active} if op == "owned-dictation-status" else {"stopped": active}))

    def _owned_speak(self, request):
        owner, key = request["owner"], (request["owner"], request["utterance"])
        signature = _signature(request)
        with self._lock:
            entries = self._owned_state()
            entry = entries.get(key)
            if entry is not None:
                if signature != entry["signature"]:
                    return protocol.reply_error(
                        "This utterance token already names different speech; "
                        "use a fresh token for each utterance.", protocol.ERR_MALFORMED)
                return protocol.reply_ok(request["id"], **self._owned_payload(entry))
            if self._speech is not None or self._arbiter.listening or self._arbiter.speaking:
                return protocol.reply_error("Audio is busy", protocol.ERR_BUSY)

        def claim(turn):
            # Runs under the lock with the turn not yet started, so the lease,
            # the owner and the retained result exist before any audio does.
            turn.owner = owner
            turn.owned_end = ""
            turn.lease_until = time.monotonic() + SPEECH_LEASE_S
            self._owned_keep(key, {"turn": turn, "token": self._owned_token(turn.id),
                                   "signature": signature})

        # Engine preparation (and a provider probe) happens outside the lock;
        # _op_speak re-checks for other audio atomically before it starts.
        reply = self._op_speak(request, claim=claim)
        if not reply.get("ok"):
            return reply
        with self._lock:
            entry = entries.get(key)
            if entry is None:   # no speakable text: nothing started
                entry = {"turn": None, "signature": signature,
                         "token": self._owned_token(self._next_turn_id("speak"))}
                self._owned_keep(key, entry)
            return protocol.reply_ok(request["id"], **self._owned_payload(entry))

    def _owned_keep(self, key, entry):
        entries = self._owned_state()
        entries[key] = entry
        while len(entries) > OWNED_RESULTS_KEPT:
            entries.popitem(last=False)

    def _owned_status(self, request):
        with self._lock:
            entry = self._owned_state().get((request["owner"], request["utterance"]))
            if entry is None:
                return protocol.reply_ok(request["id"], state="unknown", turn="",
                                         playing=False, detail="")
            if entry["turn"] is not None and entry["turn"] is self._speech:
                entry["turn"].lease_until = time.monotonic() + SPEECH_LEASE_S
            return protocol.reply_ok(request["id"], **self._owned_payload(entry))

    def _owned_stop(self, request):
        with self._lock:
            self._owned_state()
            turn = self._speech
            active = bool(turn is not None and getattr(turn, "owner", "") == request["owner"]
                          and self._owned_token(turn.id) == request["turn"])
            if active:
                self._cancel_speech()
            return protocol.reply_ok(request["id"], stopped=active)
