"""The Colab Ollama tier: session state, the tunnel endpoint, registration, commands.

- :mod:`.protocol` — the wire contract with the VM (key derivation, the
  registration signature, the tunnel URL rule, the minimum Ollama, the
  Colab models and their last verified digests);
- :mod:`.state` — the session's state document and its stores;
- :mod:`.endpoint` — :class:`~.endpoint.ColabEndpoint`, what the request
  path asks before it calls Colab (READY, busy, breaker) and the HTTP
  client for the tunnel;
- :mod:`.registration` — ``POST /internal/colab/register``;
- :mod:`.admin` — ``!colab on|off|status`` and ``!whoami``;
- :mod:`.control_plane` — :class:`~.control_plane.ColabControlPlane`, the
  async facade over the google-colab-cli library (assign, list, unassign,
  compute units, cells, uploads); Cloud Tasks and the cron tick only,
  never the request path.

See ADR-0010.
"""
