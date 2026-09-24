"""Regenerates COMMANDS.md from commands.COMMAND_REFERENCE and asserts it
matches the committed file -- catches the reference and the doc drifting
apart, rather than relying on someone remembering to regenerate by hand."""
from tools.gen_commands_doc import render_commands_markdown


def test_committed_commands_md_matches_generated_output():
    generated = render_commands_markdown()
    with open("COMMANDS.md") as f:
        committed = f.read()
    assert generated == committed


def test_command_reference_covers_new_anatomical_targets():
    committed = open("COMMANDS.md").read()
    for example in (
        "more heart",
        "high opacity vessels",
        "more liver",
        "show only kidneys",
        "brighten spleen",
    ):
        assert f"`{example}`" in committed


def test_command_reference_documents_approximate_hu_limitation():
    generated = render_commands_markdown()
    assert "approximate HU" in generated
    assert "not anatomical guarantees" in generated
