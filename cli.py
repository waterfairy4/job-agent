"""Chat with the agent in the terminal.

  python cli.py
  /job    paste a job description (finish with a line containing only END)
  /jobs   show your tracker
  /new    start a new conversation
  /quit   exit
"""
from jobagent.agent import new_thread, step
from jobagent.app_factory import create


def show(out):
    for t in out["trace"]:
        print(f"  · {t['tool']}({', '.join(f'{k}={str(v)[:40]!r}' for k, v in t['args'].items())})")
    if out["reply"]:
        print(f"\nAgent: {out['reply']}")


def read_job() -> str:
    print("Paste the job description, then a line with only END:")
    lines = []
    while (line := input()) != "END":
        lines.append(line)
    return "Here is a job description. Please analyze it and match it to my profile:\n\n" + "\n".join(lines)


def main():
    graph, index, store = create()
    thread = new_thread()
    print(f"Job agent ready ({index.size()} profile chunks). Commands: /job /jobs /new /quit")
    while True:
        text = input("\nYou: ").strip()
        if text in ("/quit", "/exit"):
            break
        if text == "/new":
            thread = new_thread()
            print("New conversation.")
            continue
        if text == "/jobs":
            for r in store.tracker():
                print(f"  {r['id']}  {str(r['score'] or '–'):>4}%  {r['status']:<10} {r['company']} – {r['title']}")
            continue
        if text == "/job":
            text = read_job()
        if not text:
            continue

        out = step(graph, thread, text)
        show(out)
        while out["pending"]:  # the agent wants to do something that needs approval
            p = out["pending"]
            ans = input(f"\nApprove: save '{p['company']} – {p['title']}' as '{p['status']}'? [y/n] ")
            out = step(graph, thread, resume=ans.strip().lower().startswith("y"))
            show(out)


if __name__ == "__main__":
    main()
