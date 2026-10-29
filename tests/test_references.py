import shutil

import pytest

from sim2sop import config, references

FOLDERS = {"c172_circuit": "air", "parallel_park": "land"}


@pytest.mark.parametrize("procedure", FOLDERS)
def test_builtin_rules_are_added_once_and_locked(data_dir, procedure):
    folder = config.REFERENCES_DIR / FOLDERS[procedure]
    added = references.sync_builtin(procedure, folder)
    assert len(added) == len(references.builtin_files(folder)) and all(d.builtin for d in added)
    assert references.active_limits(procedure)
    assert references.sync_builtin(procedure, folder) == []   # nothing changed: nothing re-added

    with pytest.raises(references.BuiltinDocument):
        references.remove_reference(procedure, added[0].id)
    with pytest.raises(references.BuiltinDocument):   # an upload with the same id cannot replace it
        references.add_reference(procedure, references.builtin_files(folder)[0])
    assert {d.id for d in references.list_references(procedure)} == {d.id for d in added}

    references.clear_references(procedure)            # Start over
    assert len(references.sync_builtin(procedure, folder)) == len(added)


def test_edited_builtin_file_is_added_again(data_dir, tmp_path):
    folder = tmp_path / "land"
    shutil.copytree(config.REFERENCES_DIR / "land", folder)
    references.sync_builtin("parallel_park", folder)
    sop = folder / "driving_test_sop.md"
    sop.write_text(sop.read_text(encoding="utf-8").replace("max: 5\n", "max: 4\n", 1), encoding="utf-8")
    assert [d.id for d in references.sync_builtin("parallel_park", folder)] == ["driving_test_sop"]
    reverse = next(lim for lim in references.active_limits("parallel_park") if lim.id == "reverse_speed_max")
    assert reverse.max == 4


def test_documents_added_before_become_builtin(data_dir):
    """A library from before built-in rules (sample documents added by hand) is taken over."""
    folder = config.REFERENCES_DIR / "air"
    for f in references.builtin_files(folder):
        references.add_reference("c172_circuit", f)
    assert len(references.sync_builtin("c172_circuit", folder)) == 3
    assert all(d.builtin for d in references.list_references("c172_circuit"))
