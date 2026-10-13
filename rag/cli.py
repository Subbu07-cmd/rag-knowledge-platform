import argparse

from rag.service import get_app_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--index", default=".index")
    a = ap.parse_args()
    out = get_app_graph(a.index).invoke({"question": a.question})
    print(out["answer"])
    if out.get("sources"):
        print("\nSources:", ", ".join(out["sources"]))


if __name__ == "__main__":
    main()
