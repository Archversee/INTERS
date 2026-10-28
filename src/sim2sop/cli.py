"""Command line: sim2sop <command> ... (run `sim2sop -h`)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from sim2sop import config, references, render, storage
from sim2sop import guidelines_store as store
from sim2sop.agent.llm import get_llm
from sim2sop.generate import generate
from sim2sop.ingest import ingest_run
from sim2sop.metrics import edits_per_version, fact_for
from sim2sop.pipeline import apply_edit, approve, create_guidelines, ingest_dir, update_guidelines
from sim2sop.profiles import list_profiles, load_profile


def _llm(args):
    if getattr(args, "provider", None) or getattr(args, "model", None):
        return get_llm(provider=args.provider, model=args.model)
    return None


def _summary(doc) -> None:
    m = doc.metrics
    check = "limit-checked" if doc.checked else "UNCHECKED"
    print(f"{doc.procedure} v{doc.version} [{doc.mode}, {doc.status}, {check}] {len(doc.steps)} guidelines "
          f"from {len(doc.runs)} runs; {len(doc.pending)} pending approval; writer {doc.writer}")
    print(json.dumps(m, indent=2))
    print(store.html_path(doc))


def cmd_generate_data(args):
    for p in generate(args.domain, n=args.runs, seed=args.seed, scenario=args.scenario):
        print(p)


def cmd_ingest(args):
    profile = load_profile(args.profile)
    for d in map(Path, args.run_dirs):
        is_parent = not (d / "meta.json").exists() and any(p.is_dir() for p in d.iterdir())
        records = ingest_dir(profile, d, args.env) if is_parent else [ingest_run(d, profile, environment=args.env)]
        for r in records:
            env = f" [{r.meta.environment}]" if r.meta.environment else ""
            print(f"{r.run_id}{env}: {len(r.events)} events, {len(r.phases)} phases, "
                  f"{sum(a.event_id is not None for a in r.annotations)}/{len(r.annotations)} annotations matched")


def cmd_add_reference(args):
    profile = load_profile(args.profile)
    for f in args.files:
        doc = references.add_reference(profile.procedure, Path(f))
        print(f"Added {doc.label} [{doc.kind}]: {len(doc.limits)} limits, {len(doc.requirements)} requirements")


def cmd_references(args):
    profile = load_profile(args.profile)
    for d in references.list_references(profile.procedure, active_only=False):
        state = "active" if d.active else "superseded"
        print(f"{d.id:16s} {d.label:50s} {d.kind:10s} {state:10s} limits={len(d.limits)} reqs={len(d.requirements)}")


def cmd_create(args):
    _summary(create_guidelines(load_profile(args.profile), llm=_llm(args)))


def cmd_update(args):
    _summary(update_guidelines(load_profile(args.profile), llm=_llm(args)))


def cmd_approve(args):
    profile = load_profile(args.profile)
    doc = approve(profile, version=args.version, approve_ids=args.steps or None, reject_ids=args.reject or [])
    print(f"v{doc.version}: {doc.status}; {len(doc.pending)} still pending")


def cmd_edit(args):
    profile = load_profile(args.profile)
    doc = apply_edit(profile, args.step_id, args.field, args.text, version=args.version)
    step = next(s for s in doc.steps if s.id == args.step_id)
    print(f"Saved as v{doc.version}; step {step.id} verified={step.verified}")
    for f in step.flags:
        print(f"  ! {f}")


def cmd_show(args):
    profile = load_profile(args.profile)
    doc = store.load(profile.procedure, args.version) if args.version else store.latest(profile.procedure)
    if doc is None:
        sys.exit("No guidelines yet: run `sim2sop create`")
    check = "limit-checked against " + "; ".join(doc.references) if doc.checked else "UNCHECKED (no reference SOPs)"
    print(f"{doc.title} — guidelines v{doc.version} ({doc.status}, {len(doc.runs)} runs) — ADVISORY, {check}")
    for phase in doc.phases:
        steps = doc.steps_in(phase)
        if steps:
            print(f"\n[{phase.upper()}]")
        for s in steps:
            tag = "" if s.mandatory else f"  (technique {s.seen_in}/{s.run_count})"
            change = f"  <{s.change}, {s.review}>" if s.change else ""
            print(f" {'✓' if s.verified else '✗'} {s.id} {s.action}{tag}{change}")
            for f, label in (("trigger", "when"), ("intent", "why"), ("typical", "typical"), ("limit", "limit")):
                if getattr(s, f):
                    print(f"      {label:8s} {getattr(s, f)}")
            for r in s.change_reasons:
                print(f"      ~ {r}")
    if doc.flags:
        print("\nFlags:")
        for f in doc.flags:
            print(f"  - [{f.kind}] {f.message}")


def cmd_diff(args):
    profile = load_profile(args.profile)
    a, b = store.load(profile.procedure, args.a), store.load(profile.procedure, args.b)
    print(json.dumps(store.diff(a, b), indent=2, ensure_ascii=False))


def cmd_export(args):
    profile = load_profile(args.profile)
    doc = store.load(profile.procedure, args.version) if args.version else store.latest(profile.procedure)
    print(render.write_html(doc))
    if args.pdf:
        print(render.write_pdf(doc))


def cmd_llm_check(args):
    llm = get_llm(provider=args.provider, model=args.model)
    print(f"provider={llm.provider} model={llm.model} base_url={llm.base_url} "
          f"api_key={'set' if llm.api_key else 'not set'}")
    if not llm.available():
        sys.exit("Not available (Ollama not running, or no API key). The template writer will be used.")
    reply = llm.chat([{"role": "user", "content": 'Reply with {"ok": true}'}], json_mode=True)
    print("reply:", reply.text.strip()[:200])


def cmd_transcribe(args):
    from sim2sop.voice import merge_into_run, transcribe
    notes = transcribe(Path(args.audio), offset_s=args.offset, model_size=args.model_size)
    print(f"{len(notes)} callouts -> {merge_into_run(Path(args.run_dir), notes)}")


def _reset(procedure: str, refs: bool, mem: bool) -> None:
    storage.delete_runs(procedure)
    shutil.rmtree(config.GUIDELINES_DIR / procedure, ignore_errors=True)
    if refs:
        references.clear_references(procedure)
    if mem:
        shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)


def cmd_reset(args):
    profile = load_profile(args.profile)
    _reset(profile.procedure, args.references, args.memory)
    print(f"Reset {profile.procedure}")


def cmd_demo(args):
    """The demo script from the brief, air domain, end to end."""
    profile = load_profile("air_c172_circuit")
    proc, llm = profile.procedure, _llm(args)
    _reset(proc, refs=True, mem=True)
    shutil.rmtree(config.RAW_DIR / "air", ignore_errors=True)
    base = generate("air", n=5)
    obstacle = generate("air", n=3, scenario="obstacle")
    ref_dir = config.REFERENCES_DIR / profile.references

    def report(doc, note):
        m = doc.metrics
        vr = fact_for(doc, "threshold:ias_kt", "trigger.ias_kt")
        print(f"v{doc.version:<2} {note:32s} {'checked' if doc.checked else 'UNCHECKED':9s} runs={len(doc.runs)} "
              f"verified={m['verifier_pass_rate']:.0%} within_limits="
              f"{'n/a' if m['within_limits'] < 0 else format(m['within_limits'], '.0%')} "
              f"range={m['range_width_ratio']:.2f} conf={m['mean_confidence']:.2f} "
              f"changed={m['steps_changed']:.0f} rotate={vr.fmt_range() if vr else '-'}")

    def publish():
        approve(profile)  # instructor approves every pending change

    print("1. Create, data only")
    ingest_run(base[0], profile)
    report(create_guidelines(profile, llm), "1 run, no references")
    publish()

    print("2. Add reference SOPs (C172 POH, aerodrome SOP, FAA handbook)")
    for f in ("c172_poh_notes.md", "aerodrome_sop_notes.md", "faa_afh_traffic_pattern_notes.md"):
        references.add_reference(proc, ref_dir / f)
    v = update_guidelines(profile, llm)
    report(v, "limits applied")
    for f in v.flags:
        if f.kind in ("clipped", "limit_violation"):
            print(f"     [{f.kind}] {f.message}")
    publish()

    print("3. More runs")
    for d in base[1:]:
        ingest_run(d, profile)
    report(update_guidelines(profile, llm), "5 runs: ranges + technique")
    publish()

    print("4. Update: NOTAM for a new obstacle + 3 runs flown around it")
    references.add_reference(proc, ref_dir / "notam_crane.md")
    for d in obstacle:
        ingest_run(d, profile)
    v = update_guidelines(profile, llm)
    report(v, "technique change")
    print(f"     phases changed: {sorted({s.phase for s in v.steps if s.change})}")
    for s in v.steps:
        if s.change:
            print(f"     {s.id} [{s.phase}] {s.label}: {'; '.join(s.change_reasons[:3])}")
    publish()

    print("5. Instructor edit, then the next generation applies it")
    flaps10 = next(s for s in v.steps if s.key == "toggle:flaps_idx:1")
    new_action = f"Select {flaps10.label}, then check the flap position indicator."
    report(apply_edit(profile, flaps10.id, "action", new_action), f"edit {flaps10.id}")
    v = update_guidelines(profile, llm)
    report(v, "correction memory")
    for s in v.steps:
        if s.change:
            print(f"     {s.id} {s.action}")
    publish()

    print("6. Other domains: land / sea adapters and profiles exist; generators are TODO")
    print(f"\nEdits per version: {edits_per_version(proc)}")
    print(f"Open: {store.html_path(store.latest(proc))}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sim2sop", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def llm_opts(p):
        p.add_argument("--provider", choices=["ollama", "openai", "anthropic", "none"],
                       help="override SIM2SOP_LLM_PROVIDER")
        p.add_argument("--model", help="override SIM2SOP_MODEL")

    p = sub.add_parser("generate-data", help="generate synthetic runs in the sim's native format")
    p.add_argument("domain", choices=["air", "land", "sea"])
    p.add_argument("-n", "--runs", type=int, default=5)
    p.add_argument("--seed", type=int)
    p.add_argument("--scenario", choices=["obstacle"], help="post-NOTAM runs around a new obstacle")
    p.set_defaults(fn=cmd_generate_data)

    p = sub.add_parser("ingest", help="ingest run folder(s) into the run library")
    p.add_argument("profile", help=f"profile name: {', '.join(list_profiles())}")
    p.add_argument("run_dirs", nargs="+", help="run folder, or a folder of run folders")
    p.add_argument("--env", help="environment tag, e.g. new-obstacle (default: from meta.json)")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("add-reference", help="add read-only reference SOPs / POH / NOTAMs")
    p.add_argument("profile")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_add_reference)

    p = sub.add_parser("references", help="list the reference library")
    p.add_argument("profile")
    p.set_defaults(fn=cmd_references)

    p = sub.add_parser("create", help="create guidelines v1 from all ingested runs")
    p.add_argument("profile")
    llm_opts(p)
    p.set_defaults(fn=cmd_create)

    p = sub.add_parser("update", help="next version: new runs, technique changes, revised references")
    p.add_argument("profile")
    llm_opts(p)
    p.set_defaults(fn=cmd_update)

    p = sub.add_parser("approve", help="approve (or reject) pending changes; publishes when none pend")
    p.add_argument("profile")
    p.add_argument("--version", type=int)
    p.add_argument("--steps", nargs="*", help="approve only these step ids (default: all pending)")
    p.add_argument("--reject", nargs="*", help="step ids to reject (revert to the previous version)")
    p.set_defaults(fn=cmd_approve)

    p = sub.add_parser("edit", help="instructor edit of one guideline field (stored in correction memory)")
    p.add_argument("profile")
    p.add_argument("step_id")
    p.add_argument("field", choices=["trigger", "action", "intent", "typical"])
    p.add_argument("text")
    p.add_argument("--version", type=int)
    p.set_defaults(fn=cmd_edit)

    p = sub.add_parser("show", help="print a guidelines version")
    p.add_argument("profile")
    p.add_argument("--version", type=int)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("diff", help="step-level diff between two versions")
    p.add_argument("profile")
    p.add_argument("a", type=int)
    p.add_argument("b", type=int)
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("export", help="write HTML (and PDF) for a version")
    p.add_argument("profile")
    p.add_argument("--version", type=int)
    p.add_argument("--pdf", action="store_true")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("llm-check", help="check the configured LLM provider responds")
    llm_opts(p)
    p.set_defaults(fn=cmd_llm_check)

    p = sub.add_parser("transcribe", help="spoken callouts (audio) -> annotations.json of a raw run")
    p.add_argument("audio")
    p.add_argument("run_dir")
    p.add_argument("--offset", type=float, default=0.0, help="telemetry time when the audio starts")
    p.add_argument("--model-size", default="small.en")
    p.set_defaults(fn=cmd_transcribe)

    p = sub.add_parser("reset", help="delete runs and guidelines for a procedure")
    p.add_argument("profile")
    p.add_argument("--references", action="store_true", help="also clear the reference library")
    p.add_argument("--memory", action="store_true", help="also clear correction memory")
    p.set_defaults(fn=cmd_reset)

    p = sub.add_parser("demo", help="run the full air demo end to end")
    llm_opts(p)
    p.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
