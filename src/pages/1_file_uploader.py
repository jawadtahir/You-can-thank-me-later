import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import base64

import streamlit as st
import hashlib
import shutil
import uuid
import json
from typing import Dict
import re
import traceback


# Constants
CHUNK_SIZE = 64 * 1024
_SAVED_UPLOADS_KEY = "saved_uploads"
_FINGERPRINTS_KEY = "fingerprints"


def _unique_dest(dest: Path) -> Path:
    """Return a non-colliding destination path by appending a counter if needed.

    Example: if "file.txt" exists, returns "file_1.txt", "file_2.txt", ...
    """
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _get_size(uploaded) -> int:
    """Try to determine the size of an uploaded file without reading it fully.

    Returns 0 when size cannot be determined.
    """
    try:
        size = getattr(uploaded, "size", None)
        if size:
            return int(size)
    except Exception:
        pass

    try:
        # attempt seek/tell
        cur = None
        try:
            cur = uploaded.tell()
        except Exception:
            cur = None
        uploaded.seek(0, os.SEEK_END)
        size = uploaded.tell()
        if cur is not None:
            try:
                uploaded.seek(cur)
            except Exception:
                pass
        else:
            try:
                uploaded.seek(0)
            except Exception:
                pass
        return int(size)
    except Exception:
        return 0


def _sanitize_rel_path(name: str) -> Path:
    """Return a safe relative Path from an uploaded name, removing absolute/parent parts."""
    rel = Path(name)
    if rel.is_absolute():
        rel = Path(rel.name)
    parts = [p for p in rel.parts if p not in ("..", "")]
    return Path(*parts) if parts else Path(rel.name)



def _stream_save_and_hash(uploaded, temp_path: Path, progress_updater) -> Tuple[str, int, Path]:
    """Stream-upload to a temporary file while computing SHA1 hash.

    Returns (sha1_hex, bytes_written, temp_path).
    """
    h = hashlib.sha1()
    written = 0
    try:
        try:
            uploaded.seek(0)
        except Exception:
            pass

        with open(temp_path, "wb") as f:
            while True:
                chunk = uploaded.read(CHUNK_SIZE)
                if not chunk:
                    break
                if isinstance(chunk, memoryview):
                    chunk = chunk.tobytes()
                f.write(chunk)
                h.update(chunk)
                written += len(chunk)
                progress_updater(len(chunk))
    except Exception:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        raise

    return h.hexdigest(), written, temp_path


def _link_or_copy(src: Path, dest: Path) -> Path:
    """Try to create a hard link from src to dest to avoid duplicating data.

    Fallbacks: symlink, then copy. Returns the path that now represents dest.
    """
    try:
        # ensure dest parent exists
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dest)
        return dest
    except Exception:
        pass
    try:
        os.symlink(src, dest)
        return dest
    except Exception:
        pass
    # final fallback: copy file (duplicates data but ensures availability)
    try:
        shutil.copy2(src, dest)
        return dest
    except Exception:
        # as a last resort, return src
        return src


def _fingerprints_file(file_dir: Path) -> Path:
    """Return the path to the fingerprints JSON file inside file_dir."""
    return file_dir / ".fingerprints.json"


def _load_fingerprints(file_dir: Path) -> Dict[str, str]:
    p = _fingerprints_file(file_dir)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k: str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_fingerprints(file_dir: Path, mapping: Dict[str, str]) -> None:
    p = _fingerprints_file(file_dir)
    tmp = p.with_name(p.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        tmp.replace(p)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _compute_sizes(files: Iterable) -> Tuple[List[int], int]:
    """Compute sizes for uploaded files when possible.

    Returns (sizes_list, total_bytes)
    """
    sizes = []
    total = 0
    for uploaded in files:
        s = _get_size(uploaded)
        sizes.append(s)
        total += s
    return sizes, total


def _read_file_bytes(path: Path) -> bytes:
    """Read a file's bytes safely. Returns empty bytes on error."""
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""


def _sanitize_key(s: str) -> str:
    """Return a widget-safe key by replacing non-alphanumeric characters with underscores."""
    # keep alphanumerics and underscore only
    return re.sub(r"[^0-9a-zA-Z_]+", "_", s)


def _widget_key(prefix: str, name: str) -> str:
    """Create a short, stable widget key using a hash of the name (avoids collisions with paths)."""
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", prefix)
    return f"{safe}_{h}"


def _delete_file_and_cleanup(file_dir: Path, rel: Path) -> bool:
    """Delete a file at file_dir/rel and clean up fingerprint and saved mappings.

    Returns True on successful deletion, False otherwise.
    """
    target = file_dir / rel
    try:
        if not target.exists() or not target.is_file():
            return False
        target.unlink()
    except Exception:
        return False

    # remove fingerprint entries that point to this path
    try:
        fprints = st.session_state.get(_FINGERPRINTS_KEY, {})
        to_remove = [k for k, v in fprints.items() if Path(v) == target]
        for k in to_remove:
            fprints.pop(k, None)
        st.session_state[_FINGERPRINTS_KEY] = fprints
        _save_fingerprints(file_dir, fprints)
    except Exception:
        pass

    # remove any saved_uploads entries that point to this path
    try:
        saved_map = st.session_state.get(_SAVED_UPLOADS_KEY, {})
        to_delete = [k for k, v in saved_map.items() if Path(v) == target]
        for k in to_delete:
            saved_map.pop(k, None)
        st.session_state[_SAVED_UPLOADS_KEY] = saved_map
    except Exception:
        pass

    # remove from selected files
    try:
        sel = set(st.session_state.get("selected_files", []))
        sel.discard(rel.as_posix())
        st.session_state["selected_files"] = list(sel)
    except Exception:
        pass

    return True


# NOTE: bulk-download-as-zip was intentionally removed per request; viewing is handled inline below.


def init_session(file_dir: Optional[str] = None) -> None:
    """Initialize session state: upload dir, saved maps and load persisted fingerprints."""
    if st.session_state.get("file_dir") is None:
        if file_dir is None:
            file_dir = os.getenv("UPLOAD_DIR", "uploads")
        st.session_state.file_dir = Path(file_dir)

    # ensure directory exists
    try:
        st.session_state.file_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        st.session_state.file_dir = Path("uploads")
        st.session_state.file_dir.mkdir(parents=True, exist_ok=True)

    # initialize saved maps
    if _SAVED_UPLOADS_KEY not in st.session_state:
        st.session_state[_SAVED_UPLOADS_KEY] = {}
    if _FINGERPRINTS_KEY not in st.session_state:
        # load persisted fingerprints (if any)
        st.session_state[_FINGERPRINTS_KEY] = _load_fingerprints(st.session_state.file_dir)
    # selected files (relative posix paths)
    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []


def render() -> None:
    """Main page renderer for file uploads.

    - Accepts directory uploads (Streamlit supports this via accept_multiple_files="directory").
    - Preserves relative directory structure under the configured upload directory.
    - Streams files to disk and updates a single overall progress bar.
    """
    st.header("Build your corpus")

    files = st.file_uploader(
        "Upload files or a directory to build your corpus",
        type=None,
        accept_multiple_files="directory",
        max_upload_size=5,
        help=(
            "Upload PDF documents, code files, or any other relevant files to build your corpus for analysis. "
            "You can upload a directory (including subdirectories) or multiple files."
        ),
    )

    # Ensure session is initialized (file_dir, saved maps, fingerprints)
    init_session()

    if not files:
        st.markdown("---")
        st.subheader("Existing files in upload directory")
        file_dir = st.session_state.file_dir


        def _human_size(n: int) -> str:
            try:
                if n is None:
                    return ""
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if n < 1024.0:
                        return f"{n:.2f}{unit}"
                    n /= 1024.0
                return f"{n:.2f}PB"
            except Exception:
                return ""

        def _render_dir(path: Path, rel: Path = Path(".")) -> None:
            """Recursively render a directory using expanders; show files with download/delete.

            This implementation uses a stable expander key derived from the relative path so
            open/closed state persists across reruns.
            """
            label = "." if rel == Path(".") else rel.name + "/"

            # create a persistent toggle to emulate an expander (some Streamlit
            # versions don't accept a 'key' argument for st.expander). We use a
            # checkbox whose state is stored in session_state so open/closed
            # state persists across reruns.
            try:
                direxp_chk = _widget_key("direxp_chk", rel.as_posix())
            except Exception:
                direxp_chk = None

            initial_expanded = (rel == Path("."))
            # Use a regular Streamlit expander (no 'key' argument) but drive its
            # initial expanded state from session_state. Some Streamlit builds do
            # not accept a 'key' argument on st.expander; this keeps the visual
            # expander while aiming to preserve a stored preference.
            if direxp_chk and direxp_chk not in st.session_state:
                try:
                    st.session_state[direxp_chk] = initial_expanded
                except Exception:
                    pass
            expanded = st.session_state.get(direxp_chk, initial_expanded)

            # Note: without a widget key on st.expander we can't directly be
            # notified when the user toggles it, so the session_state value may
            # not reflect manual toggles. This approach restores the native
            # expander UI while using session_state for initial state.
            try:
                expander_ctx = st.expander(label, expanded=expanded)
            except TypeError:
                # some Streamlit builds may still reject 'expanded' kwarg; fall
                # back to an unkeyed expander without explicit expanded state.
                expander_ctx = st.expander(label)

            with expander_ctx:
                # list files in this directory
                try:
                    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
                except Exception:
                    st.write("(unable to read directory)")
                    return

                for entry in entries:
                    entry_rel = (rel / entry.name) if rel != Path(".") else Path(entry.name)
                    if entry.is_dir():
                        _render_dir(entry, entry_rel)
                        continue

                    cols = st.columns([1, 4, 1, 1])

                    # checkbox column for bulk selection
                    sel_key = _widget_key("chk", entry_rel.as_posix())
                    selected_set = set(st.session_state.get("selected_files", []))
                    initial_checked = entry_rel.as_posix() in selected_set
                    if sel_key not in st.session_state:
                        try:
                            st.session_state[sel_key] = initial_checked
                        except Exception:
                            pass

                    # render checkbox using the persistent key (no value param when key is provided)
                    try:
                        val = cols[0].checkbox(" ", key=sel_key, label_visibility="collapsed")
                    except Exception:
                        val = cols[0].checkbox(" ", value=initial_checked, label_visibility="collapsed")

                    # synchronize the global selected_files list with the checkbox state
                    if val and (entry_rel.as_posix() not in selected_set):
                        selected_set.add(entry_rel.as_posix())
                    if (not val) and (entry_rel.as_posix() in selected_set):
                        selected_set.discard(entry_rel.as_posix())

                    try:
                        size = entry.stat().st_size
                    except Exception:
                        size = 0
                    cols[1].write(f"{entry_rel.name} — {_human_size(size)}")

                    # view button (inline preview for common types)
                    try:
                        data = _read_file_bytes(entry)
                        if not data:
                            cols[2].write("")
                        else:
                            view_key = _widget_key("view", entry_rel.as_posix())
                            exp_key = _widget_key("exp", entry_rel.as_posix())
                            if cols[2].button("View", key=view_key):
                                ext = entry.suffix.lower()
                                with st.expander(f"Preview: {entry_rel.as_posix()}", expanded=True, key=exp_key):
                                    # images
                                    if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
                                        try:
                                            st.image(data, caption=entry.name)
                                        except Exception:
                                            st.write("(unable to render image)")
                                    # text-like files
                                    elif ext in (".txt", ".py", ".md", ".json", ".csv", ".log"):
                                        try:
                                            text = data.decode("utf-8")
                                        except Exception:
                                            text = data.decode("latin-1", errors="replace")
                                        st.code(text)
                                    # PDFs: render in iframe via data URL
                                    elif ext == ".pdf":
                                        try:
                                            b64 = base64.b64encode(data).decode("utf-8")
                                            pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600px"></iframe>'
                                            st.components.v1.html(pdf_display, height=600)
                                        except Exception:
                                            st.write("(unable to render PDF)")
                                    else:
                                        st.write(f"No preview available for *{ext}* files.")
                    except Exception:
                        cols[2].write("")

                    # delete button (use helper)
                    del_key = _widget_key("del", entry_rel.as_posix())
                    if cols[3].button("Delete", key=del_key):
                        ok = _delete_file_and_cleanup(st.session_state.file_dir, entry_rel)
                        if not ok:
                            st.error(f"Failed to delete {entry_rel}")
                        else:
                            st.rerun(scope="app")

        # Render the upload dir as a tree
        try:
            if not file_dir.exists():
                st.write("(no files yet)")
            else:
                # ensure the authoritative selected_files list matches per-checkbox widget state
                def _recompute_selected_from_widget_keys() -> None:
                    try:
                        sel = []
                        for r, ds, fs in os.walk(file_dir):
                            rp = Path(r)
                            for fn in fs:
                                rel = rp.relative_to(file_dir) / fn if rp != file_dir else Path(fn)
                                key = _widget_key("chk", rel.as_posix())
                                if st.session_state.get(key):
                                    sel.append(rel.as_posix())
                        st.session_state["selected_files"] = sel
                    except Exception:
                        # leave existing selected_files unchanged on failure
                        pass

                _recompute_selected_from_widget_keys()

                # bulk controls
                cols = st.columns([1, 1, 1, 4])
                # select all
                if cols[0].button("Select all"):
                    all_files = []
                    for r, ds, fs in os.walk(file_dir):
                        rp = Path(r)
                        for fn in fs:
                            rel = rp.relative_to(file_dir) / fn if rp != file_dir else Path(fn)
                            rp_str = rel.as_posix()
                            all_files.append(rp_str)
                            # set individual checkbox states so UI reflects selection
                            chk_key = _widget_key("chk", rp_str)
                            try:
                                st.session_state[chk_key] = True
                            except Exception:
                                pass
                    st.session_state["selected_files"] = all_files
                    #st.experimental_rerun()

                # clear selection
                if cols[1].button("Clear selection"):
                    # clear selected_files and reset per-checkbox states
                    try:
                        all_keys = []
                        for r, ds, fs in os.walk(file_dir):
                            rp = Path(r)
                            for fn in fs:
                                rel = rp.relative_to(file_dir) / fn if rp != file_dir else Path(fn)
                                all_keys.append(_widget_key("chk", rel.as_posix()))
                        for k in all_keys:
                            try:
                                st.session_state[k] = False
                            except Exception:
                                pass
                    except Exception:
                        pass
                    st.session_state["selected_files"] = []
                    #st.experimental_rerun()

                # delete selected
                if cols[2].button("Delete selected"):
                    sel = list(st.session_state.get("selected_files", []))
                    for rp in sel:
                        try:
                            _delete_file_and_cleanup(file_dir, Path(rp))
                        except Exception:
                            # ignore individual failures, continue with others
                            pass
                    st.session_state["selected_files"] = []
                    st.rerun(scope="app")

                # show selection count (bulk download removed)
                try:
                    sel = list(st.session_state.get("selected_files", []))
                    if sel:
                        cols[3].write(f"{len(sel)} selected")
                    else:
                        cols[3].write("")
                except Exception:
                    cols[3].write("")

                try:
                    _render_dir(file_dir, Path("."))
                except Exception as e:
                    # show full traceback in the UI to aid debugging
                    tb = traceback.format_exc()
                    st.markdown("**DEBUG: exception while rendering upload tree**")
                    st.text(str(e))
                    st.text(tb)
        except Exception:
            st.write("(no files yet)")
        return

    sizes, total_bytes = _compute_sizes(files)
    overall_pb = st.progress(0)
    bytes_written = 0

    def _update_progress(delta: int) -> None:
        nonlocal bytes_written
        bytes_written += delta
        try:
            if total_bytes > 0:
                overall_pb.progress(int(bytes_written / total_bytes * 100))
            else:
                # when sizes unknown, move progress by files written roughly; handled in loop as fallback
                overall_pb.progress(min(100, int(bytes_written / (1024 * 1024) * 2)))
        except Exception:
            pass

    saved: List[Path] = []
    for idx, uploaded in enumerate(files):
        status = st.empty()
        status.text(f"Saving {uploaded.name}...")

        rel = _sanitize_rel_path(uploaded.name)
        size = sizes[idx] if idx < len(sizes) else _get_size(uploaded)
        key = f"{rel.as_posix()}:{size}"

        # If we've already saved this uploaded file previously, reuse recorded path and skip saving.
        prev = st.session_state[_SAVED_UPLOADS_KEY].get(key)
        if prev:
            prev_path = Path(prev)
            if prev_path.exists():
                status.text(f"Already saved: {uploaded.name}")
                saved.append(prev_path)
                # ensure overall progress reflects this file
                if total_bytes == 0:
                    overall_pb.progress(int((idx + 1) / len(files) * 100))
                else:
                    # increase bytes_written by the known size to keep progress consistent
                    try:
                        bytes_written += size
                        overall_pb.progress(int(bytes_written / total_bytes * 100))
                    except Exception:
                        pass
                continue

        dest = st.session_state.file_dir / rel
        # avoid overwriting existing files
        dest = _unique_dest(dest)

        # write to a temp file and compute hash while streaming
        temp_name = f".tmp-{uuid.uuid4().hex}"
        temp_path = dest.parent / temp_name
        try:
            sha, written, _ = _stream_save_and_hash(uploaded, temp_path, _update_progress)
            # if we already have this content saved somewhere, link/copy instead of keeping duplicate
            existing = st.session_state[_FINGERPRINTS_KEY].get(sha)
            if existing and Path(existing).exists():
                # link to existing file to avoid data duplication
                new_path = _link_or_copy(Path(existing), dest)
                # remove the temp file
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                saved.append(new_path)
                status.text(f"Saved {uploaded.name} (deduplicated)")
                # record by rel:size as well for quick skip
                try:
                    st.session_state[_SAVED_UPLOADS_KEY][key] = str(new_path)
                except Exception:
                    pass
            else:
                # move temp file into final destination
                try:
                    temp_path.replace(dest)
                    saved_path = dest
                except Exception:
                    shutil.move(str(temp_path), str(dest))
                    saved_path = dest
                # record fingerprint -> path
                try:
                    st.session_state[_FINGERPRINTS_KEY][sha] = str(saved_path)
                    # persist fingerprints to disk immediately
                    _save_fingerprints(st.session_state.file_dir, st.session_state[_FINGERPRINTS_KEY])
                except Exception:
                    pass
                saved.append(saved_path)
                status.text(f"Saved {uploaded.name}")
                try:
                    st.session_state[_SAVED_UPLOADS_KEY][key] = str(saved_path)
                except Exception:
                    pass

            # If sizes were unknown, nudge the progress bar by file-count fraction
            if total_bytes == 0:
                overall_pb.progress(int((idx + 1) / len(files) * 100))
        except Exception as exc:
            # cleanup temp file if any
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            status.error(f"Failed to save {uploaded.name}: {exc}")

    if saved:
        st.success(f"Saved {len(saved)} file(s) to '{st.session_state.file_dir}'")
        for p in saved:
            try:
                st.write(str(p.relative_to(st.session_state.file_dir)))
            except Exception:
                st.write(p.name)



init_session()
render()

# if __name__ == "__main__":
#     # allow running this page file directly for quick testing
#     init_session()
#     render()
