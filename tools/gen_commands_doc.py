"""Generates COMMANDS.md from commands.COMMAND_REFERENCE -- the one source
of truth also served by server.py's /api/commands endpoint for the in-app
help panel. Run this script whenever COMMAND_REFERENCE changes."""
from commands import COMMAND_REFERENCE


def render_commands_markdown() -> str:
    lines = ["# Supported Commands", ""]
    lines.append(
        "Generated from `commands.COMMAND_REFERENCE` -- run "
        "`python -m tools.gen_commands_doc` after changing it."
    )
    lines.append("")
    for entry in COMMAND_REFERENCE:
        lines.append(f"## {entry['category']}")
        lines.append("")
        lines.append(entry["description"])
        lines.append("")
        for example in entry["examples"]:
            lines.append(f"- `{example}`")
        lines.append("")
    return "\n".join(lines)


def main():
    content = render_commands_markdown()
    with open("COMMANDS.md", "w") as f:
        f.write(content)
    print("Wrote COMMANDS.md")


if __name__ == "__main__":
    main()
