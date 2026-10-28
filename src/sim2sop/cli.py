"""Command line: sim2sop <command> ... (run `sim2sop -h`)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from sim2sop import config, render, sop_store, storage
from sim2sop.agent.llm import get_llm
from sim2sop.generate import generate
from sim2sop.ingest import ingest_run
from sim2sop.metrics import band_for, edits_per_version
from sim2sop.pipeline import apply_edit, generate_sop, ingest_dir
from sim2sop.profiles import list_profiles, load_profile


def _llm(args):
    if getattr(args, "provider", None) or getattr(args, "model", None):
        return get_llm(provider=args.provider, model=args.model)
    return None


def cmd_generate_data(args):
    paths = generate(args.domain, n=args.runs, seed=args.seed)
    for p in paths:
        print(p)


def cmd_ingest(args):
    profile = load_profile(args.profile)
    for d in args.run_dirs:
        d = Path(d)
        records = ingest_dir(profile, d) if not (d / "meta.json").exists() and any(
            p.is_dir() for p in d.iterdir()) else [ingest_run(d, profile)]
        for r in records:
            print(f"{r.run_id}: {len(r.events)} events, {len(r.phases)} phases, "
                  f"{sum(a.event_id is not None for a in r.annotations)}/{len(r.annotations)} annotations matched, "
                  f"{len(r.flags)} flags")


def cmd_build(args):
    profile = load_profile(args.profile)
    sop = generate_sop(profile, llm=_llm(args), incremental=not args.full)
    print(f"{profile.procedure} v{sop.version} ({sop.writer}): {len(sop.steps)} steps")
    print(json.dumps(sop.metrics, indent=2))
    print(config.SOPS_DIR / profile.procedure / f"v{sop.version}.html")


def cmd_edit(args):
    profile = load_profile(args.profile)
    sop = apply_edit(profile, args.step_id, args.field, args.text, version=args.version)
    step = next(s for s in sop.steps if s.id == args.step_id)
    print(f"Saved as v{sop.version}; step {step.id} verified={step.verified}")
    for f in step.flags:
        print(f"  ! {f}")


def cmd_show(args):
    profile = load_profile(args.profile)
    sop = sop_store.load(profile.procedure, args.version) if args.version else sop_store.latest(profile.procedure)
    if sop is None:
        sys.exit("No SOP yet: run `sim2sop build`")
    print(f"{sop.title} — v{sop.version} ({len(sop.runs)} runs, writer {sop.writer})")
    for phase in sop.phases:
        steps = sop.steps_in(phase)
        if steps:
            print(f"\n[{phase.upper()}]")
        for s in steps:
            tag = "" if s.mandatory else f"  (technique {s.seen_in}/{s.run_count})"
            ok = "✓" if s.verified else "✗"
            print(f" {ok} {s.id} {s.action}{tag}")
            for f in ("trigger", "intent", "expected", "tolerance"):
                if getattr(s, f):
                    print(f"      {f:9s} {getattr(s, f)}")
    if sop.flags:
        print("\nFlags:")
        for f in sop.flags:
            print(f"  - [{f.kind}] {f.message}")


def cmd_diff(args):
    profile = load_profile(args.profile)
    a, b = sop_store.load(profile.procedure, args.a), sop_store.load(profile.procedure, args.b)
    print(json.dumps(sop_store.diff(a, b), indent=2, ensure_ascii=False))


def cmd_export(args):
    profile = load_profile(args.profile)
    sop = sop_store.load(profile.procedure, args.version) if args.version else sop_store.latest(profile.procedure)
    print(render.write_html(sop))
    if args.pdf:
        print(render.write_pdf(sop))


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


def cmd_reset(args):
    profile = load_profile(args.profile)
    storage.delete_runs(profile.procedure)
    shutil.rmtree(config.SOPS_DIR / profile.procedure, ignore_errors=True)
    if args.memory:
        shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)
    print(f"Reset {profile.procedure}")


def cmd_demo(args):
    """The 5-minute demo: v1 from one run, v2 from five, instructor edit, v4 learns it."""
    profile = load_profile("air_c172_circuit")
    llm = _llm(args)
    storage.delete_runs(profile.procedure)
    shutil.rmtree(config.SOPS_DIR / profile.procedure, ignore_errors=True)
    shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)
    raws = generate("air", n=5, seed=args.seed)

    def report(sop, note):
        m = sop.metrics
        vr = band_for(sop, "threshold:ias_kt", "trigger.ias_kt")
        vr_txt = f"Vr {vr[0]:.0f} ±{vr[1]:.1f} kt" if vr else ""
        print(f"v{sop.version:<2} {note:34s} runs={len(sop.runs)} steps={len(sop.steps):<3} "
              f"verified={m['verifier_pass_rate']:.0%} cited={m['grounded_pct']:.0%} "
              f"band={m['band_width_ratio']:.2f} conf={m['mean_confidence']:.2f} "
              f"technique={m['technique_steps']:.0f} {vr_txt}")

    ingest_run(raws[0], profile)
    report(generate_sop(profile, llm=llm), "1 run")
    for r in raws[1:]:
        ingest_run(r, profile)
    v2 = generate_sop(profile, llm=llm)
    report(v2, "5 runs: consensus + bands")

    flaps10 = next(s for s in v2.steps if s.key == "toggle:flaps_idx:1")
    new_action = f"Select {flaps10.label}, then check the flap position indicator."
    v3 = apply_edit(profile, flaps10.id, "action", new_action)
    report(v3, f"instructor edits {flaps10.id} action")
    print(f"     before: {flaps10.action!r}\n     after:  {new_action!r}")

    v4 = generate_sop(profile, llm=llm)
    report(v4, "regenerated with correction memory")
    for s in v4.steps:
        if s.signal == "flaps_idx" and s.kind == "toggle":
            print(f"     {s.id} {s.action}")
    print(f"\nEdits per version: {edits_per_version(profile.procedure)}")
    print(f"Open: {config.SOPS_DIR / profile.procedure / f'v{v4.version}.html'}")


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
    p.add_argument("--seed", type=int, default=172)
    p.set_defaults(fn=cmd_generate_data)

    p = sub.add_parser("ingest", help="ingest run folder(s) into the run library")
    p.add_argument("profile", help=f"profile name: {', '.join(list_profiles())}")
    p.add_argument("run_dirs", nargs="+", help="run folder, or a folder of run folders")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("build", help="generate the next SOP version from all ingested runs")
    p.add_argument("profile")
    p.add_argument("--full", action="store_true", help="rewrite every step, not only changed ones")
    llm_opts(p)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("edit", help="instructor edit of one step field (stored in correction memory)")
    p.add_argument("profile")
    p.add_argument("step_id")
    p.add_argument("field", choices=["trigger", "action", "intent", "expected", "tolerance"])
    p.add_argument("text")
    p.add_argument("--version", type=int)
    p.set_defaults(fn=cmd_edit)

    p = sub.add_parser("show", help="print an SOP version")
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

    p = sub.add_parser("reset", help="delete runs and SOPs for a procedure")
    p.add_argument("profile")
    p.add_argument("--memory", action="store_true", help="also clear correction memory")
    p.set_defaults(fn=cmd_reset)

    p = sub.add_parser("demo", help="run the full air demo end to end")
    p.add_argument("--seed", type=int, default=172)
    llm_opts(p)
    p.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
