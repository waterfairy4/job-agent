"""Write the API description (OpenAPI) that Dify or Coze import to use Job Agent as a tool.

  python export_openapi.py                                   -> integrations/openapi.json
  python export_openapi.py http://host.docker.internal:8000  (the address Dify can reach; this is the default)
  python export_openapi.py https://your-tunnel.example.com   (a public HTTPS address, needed for Coze)

No model or API key is needed: only the routes are described.
"""
import json
import sys
from pathlib import Path

from jobagent.api import create_app


def main():
    server = sys.argv[1] if len(sys.argv) > 1 else "http://host.docker.internal:8000"
    spec = create_app(graph=None, store=None).openapi()
    spec["servers"] = [{"url": server}]
    out = Path("integrations") / "openapi.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(spec['paths'])} endpoints, server {server})")


if __name__ == "__main__":
    main()
