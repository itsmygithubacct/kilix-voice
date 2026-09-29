"""Opt-in owned Avatar turns layered on the daemon's normal job lifecycle."""
import hashlib
import time
from collections import OrderedDict
from . import protocol

class OwnedTurns:
    def _owned_payload(self, entry):
        turn=entry['turn']
        outcome=turn.outcome if turn else None
        state=('completed' if turn is None else outcome.outcome if outcome else
               'speaking' if getattr(self,'_player',None) and self._player.playing else 'preparing')
        if state=='deadline': state='expired'
        return dict(turn=turn.id if turn else '',state=state,playing=state=='speaking',detail=outcome.message if outcome else '')

    def _owned_expire(self):
        with self._lock:
            for turn in (self._speech,self._dictation):
                if turn is not None and getattr(turn,'owner','') and time.monotonic()>=turn.lease_until:
                    if turn is self._speech: self._cancel_speech()
                    else:
                        turn.abort=True
                        turn.stop.set()

    def _op_owned(self, request):
        with self._lock:
            if not hasattr(self,'_owned_entries'): self._owned_entries=OrderedDict()
            op=request['op']; owner=request['owner']
            if op in ('owned-speak','owned-status'):
                key=(owner,request['utterance'])
                entry=self._owned_entries.get(key)
                if entry:
                    if op=='owned-speak':
                        signature=hashlib.sha256(protocol.encode({k:request[k] for k in ('text','model','voice','rate') if k in request})).hexdigest()
                        if signature!=entry['signature']:
                            return protocol.reply_error('Utterance token already names different speech')
                    if entry['turn']: entry['turn'].lease_until=time.monotonic()+5
                    return protocol.reply_ok(request['id'],**self._owned_payload(entry))
                if op=='owned-status': return protocol.reply_ok(request['id'],state='unknown',turn='')
                if self._arbiter.listening or self._arbiter.speaking:
                    return protocol.reply_error('Audio is busy',protocol.ERR_BUSY)
                reply=self._op_speak(request)
                if not reply.get('ok'): return reply
                turn=self._speech
                if turn:
                    turn.owner=owner;turn.lease_until=time.monotonic()+5
                signature=hashlib.sha256(protocol.encode({k:request[k] for k in ('text','model','voice','rate') if k in request})).hexdigest()
                entry={'turn':turn,'signature':signature}
                self._owned_entries[key]=entry
                while len(self._owned_entries)>128: self._owned_entries.popitem(last=False)
                return protocol.reply_ok(request['id'],**self._owned_payload(entry))
            if op=='owned-stop':
                turn=self._speech
                active=bool(turn and getattr(turn,'owner','')==owner and turn.id==request['turn'])
                if active: self._cancel_speech()
                return protocol.reply_ok(request['id'],stopped=active)
            if op=='owned-dictate':
                if self._arbiter.speaking:
                    return protocol.reply_error('Audio is busy',protocol.ERR_BUSY)
                return self._op_dictate(request)
            turn=self._dictation
            active=bool(turn and getattr(turn,'owner','')==owner)
            if active:
                if op=='owned-dictation-status': turn.lease_until=time.monotonic()+3
                else:
                    with turn.capture_lock:
                        turn.owned_finish.set()
                        if turn.capture is not None:
                            turn.capture.request_stop()
            return protocol.reply_ok(request['id'],**({'active':active} if op=='owned-dictation-status' else {'stopped':active}))
