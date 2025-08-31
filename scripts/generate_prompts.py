import re
from pathlib import Path

TEMPLATE_PATH = Path("001_layers_baseline/prompt-single-model-evaluation.txt")
RUN_LATEST_DIR = Path("001_layers_baseline/run-latest")
OUTPUT_DIR = Path("001_layers_baseline/prompts")


def load_template() -> str:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    # Ensure the SCRIPT path is present just after its label.
    script_marker = "- SCRIPT – source code of the probe's script (for context):"
    if script_marker in text and "001_layers_baseline/run.py" not in text:
        text = text.replace(
            script_marker + " \n",
            script_marker + " \n001_layers_baseline/run.py\n",
        )
    return text


def iter_model_ids():
    # Model IDs are derived from files named output-<model>.json in run-latest
    for json_path in sorted(RUN_LATEST_DIR.glob("output-*.json")):
        name = json_path.name  # e.g., output-Meta-Llama-3-8B.json
        if not name.startswith("output-") or not name.endswith(".json"):
            continue
        model_id = name[len("output-") : -len(".json")]
        yield model_id


def build_prompt(template: str, model_id: str) -> str:
    # Replace MODEL placeholders with the model_id
    # Only the 'output-MODEL' tokens appear in the template; replace conservatively.
    content = template.replace("output-MODEL", f"output-{model_id}")
    # Inject explicit destination for the EVAL output markdown file under INPUTS.
    anchor = "\n- Your own research knowledge.\n"
    insertion = (
        f"\n- Your own research knowledge.\n\n"
        f"- EVAL output file: 001_layers_baseline/run-latest/evaluation-{model_id}.md\n"
    )
    if anchor in content and f"evaluation-{model_id}.md" not in content:
        content = content.replace(anchor, insertion)
    return content


def main():
    template = load_template()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for model_id in iter_model_ids():
        prompt = build_prompt(template, model_id)
        out_path = OUTPUT_DIR / f"prompt-evaluation-{model_id}.md"
        out_path.write_text(prompt, encoding="utf-8")
        count += 1

    print(f"Wrote {count} prompt(s) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
