import tempfile
import unittest
from pathlib import Path

from optiverse import codebase


class TemporaryDirectoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        codebase.make_writable(self.root)
        self._temporary_directory.cleanup()

    def write(self, relative_path: str, content: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path


class RelativeFilesTest(TemporaryDirectoryTestCase):
    def test_lists_nested_files_as_sorted_posix_paths(self) -> None:
        self.write("b.py", "b")
        self.write("a/deep/c.txt", "c")
        self.write("a/b.py", "ab")

        self.assertEqual(
            codebase.relative_files(self.root), ["a/b.py", "a/deep/c.txt", "b.py"]
        )

    def test_empty_directory_has_no_files(self) -> None:
        self.assertEqual(codebase.relative_files(self.root), [])


class MaterializeTest(TemporaryDirectoryTestCase):
    def test_round_trips_a_nested_tree(self) -> None:
        self.write("source/main.go", "package main")
        self.write("source/pkg/util.go", "package main // util")

        destination = self.root / "destination"
        codebase.materialize(self.root / "source", destination)

        self.assertEqual(
            codebase.relative_files(destination), ["main.go", "pkg/util.go"]
        )
        self.assertEqual(
            (destination / "pkg" / "util.go").read_text(), "package main // util"
        )

    def test_copy_of_a_read_only_tree_is_writable(self) -> None:
        """The store is read-only and copytree preserves modes, so a naive copy
        would hand the agent a directory it cannot edit."""
        self.write("source/solver.py", "original")
        source = self.root / "source"
        codebase.make_read_only(source)

        destination = self.root / "destination"
        codebase.materialize(source, destination)

        (destination / "solver.py").write_text("edited")
        self.assertEqual((destination / "solver.py").read_text(), "edited")

    def test_source_is_untouched_by_edits_to_the_copy(self) -> None:
        self.write("source/solver.py", "original")
        source = self.root / "source"
        codebase.make_read_only(source)

        destination = self.root / "destination"
        codebase.materialize(source, destination)
        (destination / "solver.py").write_text("edited")

        self.assertEqual((source / "solver.py").read_text(), "original")


class MakeReadOnlyTest(TemporaryDirectoryTestCase):
    def test_blocks_writes_to_existing_files(self) -> None:
        path = self.write("tree/solver.py", "original")
        codebase.make_read_only(self.root / "tree")

        with self.assertRaises(PermissionError):
            path.write_text("edited")

    def test_make_writable_restores_access(self) -> None:
        path = self.write("tree/solver.py", "original")
        tree = self.root / "tree"

        codebase.make_read_only(tree)
        codebase.make_writable(tree)

        path.write_text("edited")
        self.assertEqual(path.read_text(), "edited")


class DigestTest(TemporaryDirectoryTestCase):
    def test_identical_trees_share_a_digest(self) -> None:
        self.write("a/one.txt", "hello")
        self.write("b/one.txt", "hello")

        self.assertEqual(
            codebase.digest(self.root / "a"), codebase.digest(self.root / "b")
        )

    def test_changed_content_changes_the_digest(self) -> None:
        path = self.write("tree/one.txt", "hello")
        before = codebase.digest(self.root / "tree")

        path.write_text("goodbye")

        self.assertNotEqual(before, codebase.digest(self.root / "tree"))

    def test_added_file_changes_the_digest(self) -> None:
        self.write("tree/one.txt", "hello")
        before = codebase.digest(self.root / "tree")

        self.write("tree/two.txt", "")

        self.assertNotEqual(before, codebase.digest(self.root / "tree"))

    def test_renamed_file_changes_the_digest(self) -> None:
        """Content alone is not enough: paths are part of the codebase."""
        path = self.write("tree/one.txt", "hello")
        before = codebase.digest(self.root / "tree")

        path.rename(self.root / "tree" / "two.txt")

        self.assertNotEqual(before, codebase.digest(self.root / "tree"))


class RemoveTreeTest(TemporaryDirectoryTestCase):
    def test_removes_a_read_only_tree(self) -> None:
        self.write("tree/one.txt", "hello")
        tree = self.root / "tree"
        codebase.make_read_only(tree)

        codebase.remove_tree(tree)

        self.assertFalse(tree.exists())

    def test_missing_tree_is_not_an_error(self) -> None:
        codebase.remove_tree(self.root / "absent")


if __name__ == "__main__":
    unittest.main()
