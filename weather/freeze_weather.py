"""
Freeze the weather study (analysis plan, Section 10, step 2): hash the processing, its inputs and the environments,
rerun every test on the final code, then lock, commit, tag `weather-plan` in both repositories and publish the plan and
the lock. Nothing here makes a forecast with real weather or computes a weather-forecast error.

Checks, in order (any failure refuses the freeze and changes nothing):
  1  Both repositories clean, equal to GitHub, and without the tag.
  2  The full backtest build (16 March 2024 to 31 October 2026): every day built or dropped with its reason, built by
     today's processing code (the hashes build.json recorded equal the files now), its outputs equal to build.json.
  3  The decision-13 sensitivity inputs (20 and 45 days) built.
  4  Spatial weights equal to the agreed hash; model weights equal to the demand study's lock.
  5  On the final code: the daily push's self-test (W1-W11), the synthetic pipeline (P1-P5), the models' self-test,
     the synthetic end-to-end scoring, and the dry gates with corruption tests on the full build.
Then writes WEATHER_LOCK.json (private repository, read by push_daily.py and run_weather.py), commits it, copies the
plan and the lock to the public repository's weather_study/, commits, tags both `weather-plan`, pushes, and reports
success only when GitHub shows both commits and both tags.

Run in `gbwx`, from anywhere, after the full build and sensitivity_inputs.py:
    python weather/freeze_weather.py          checks and tests only; prints the lock it would write
    python weather/freeze_weather.py --tag    freezes
"""
import datetime as dt, glob, json, os, platform, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PRIVATE = os.path.dirname(HERE)
PUBLIC = os.path.expanduser(os.environ.get('GBFM_PUBLIC', '~/gb-foundation-model-tests'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(PUBLIC, 'code'))
import sources as S

TAG = 'weather-plan'
DATA = os.path.expanduser('~/gb-weather-data'); BT = os.path.join(DATA, 'backtest')
FULL = '2024-03-16_2026-10-31'; N_DAYS = 960
SENS = [f'min{m}_2024-03-16_2024-05-31' for m in (20, 45)]
WEIGHTS = os.path.join(DATA, 'weights', 'weights.nc')
WEIGHTS_SHA = 'c3af7721b673386ef31285ff55deab71617bff774b7cb5c13e448dee543102be'
ENVS = {'wx': '~/miniconda3/envs/wx/bin/python', 'gbwx': '~/miniconda3/envs/gbwx/bin/python'}
SEEDS = dict(backtest=20261120, prospective=20270401, holidays=20261121, power=20261108)
PUBLIC_DIR = 'weather_study'


def tilde(p):
    home = os.path.expanduser('~'); return '~' + p[len(home):] if p.startswith(home) else p


def git(repo, *a):
    r = subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def fail(msg): sys.exit('Freeze refused: ' + msg)


def say(*a): print(f'{dt.datetime.now():%H:%M:%S}', *a, flush=True)


def repo_ready(repo):
    git(repo, 'fetch', '-q', 'origin', '--tags')
    if git(repo, 'status', '--porcelain')[1]: fail(f'{repo}: uncommitted changes:\n' + git(repo, 'status', '--short')[1])
    if git(repo, 'rev-parse', 'HEAD')[1] != git(repo, 'rev-parse', '@{u}')[1]: fail(f'{repo}: differs from GitHub; pull or push first')
    if TAG in git(repo, 'ls-remote', '--tags', 'origin', TAG)[1]: fail(f'{repo}: tag {TAG} already exists on GitHub')


def run(label, cmd, cwd=PRIVATE, env=None):
    say(f'test: {label}')
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    lines = [l for l in (r.stdout + r.stderr).splitlines() if 'Warning' not in l and l.strip()]
    if r.returncode or any(' FAIL ' in f' {l} ' for l in lines):
        fail(f'{label} failed (exit {r.returncode}):\n' + '\n'.join(lines[-25:]))
    return [l for l in lines if 'PASS' in l]                  # recorded in the lock


def checks():
    # 1 repositories
    for repo in (PRIVATE, PUBLIC): repo_ready(repo)
    # 2 the full build, from today's code
    b = json.load(open(os.path.join(BT, f'build_{FULL}.json')))
    if b['days_built'] + b['days_dropped'] != N_DAYS: fail(f'build covers {b["days_built"] + b["days_dropped"]} days, not {N_DAYS}')
    for f, h in b['code_sha256'].items():
        if S.sha256(os.path.join(HERE, f)) != h: fail(f'{f} changed since the backtest was built; rebuild')
    for f, h in b['outputs'].items():
        if S.sha256(os.path.join(BT, f)) != h: fail(f'{f} differs from build.json')
    # 3 sensitivity inputs
    for t in SENS:
        for f in (f'inputs_{t}.csv.gz', f'days_{t}.jsonl'):
            if not os.path.exists(os.path.join(BT, f)): fail(f'{f} missing: run weather/sensitivity_inputs.py')
    # 4 weights
    if S.sha256(WEIGHTS) != WEIGHTS_SHA: fail('weights.nc differs from the agreed hash')
    import models as M
    study = json.load(open(os.path.join(PUBLIC, 'plan', 'PLAN_LOCK.json')))
    from huggingface_hub import hf_hub_download
    for name, (repo, rev) in M.REVISIONS.items():
        if S.sha256(hf_hub_download(repo, 'model.safetensors', revision=rev)) != study['weights_sha256'][name]:
            fail(f'{name} weights differ from the demand study lock')
    # 5 tests on the final code
    py = sys.executable; tests = {}
    tests['push_daily_self_test'] = run('daily push self-test', [py, 'weather/push_daily.py', '--self-test'])
    run('synthetic pipeline P1-P5', [py, os.path.join(HERE, 'synth_test.py'), 'pipeline'], cwd=PUBLIC, env={'JAX_PLATFORMS': 'cpu'})
    pj = json.load(open(os.path.join(DATA, 'synth', 'pipeline.json')))
    if not pj['pipeline']['pass']: fail('synthetic pipeline did not pass')
    tests['synth_pipeline'] = {k: v for k, v in pj['pipeline'].items() if k.startswith('P') or k == 'pass'}
    tests['models_selftest'] = run('models self-test', [py, 'weather/run_weather.py', 'models-selftest'], env={'JAX_PLATFORMS': 'cpu'})
    tests['synthetic_scoring'] = run('synthetic scoring', [py, 'weather/run_weather.py', 'synthetic'])
    tests['dry_gates'] = run('dry gates with corruption tests', [py, 'weather/run_weather.py', 'dry', '--corrupt'])

    code = sorted(glob.glob(os.path.join(HERE, '*.py'))) + sorted(glob.glob(os.path.join(PRIVATE, 'feasibility', '*.py'))) + \
        [os.path.join(PRIVATE, 'docs', f) for f in ('PLAN_weather.md', 'BRIEF_4_weather.md')]
    files = [os.path.join(BT, f) for f in b['outputs']] + [os.path.join(BT, f'build_{FULL}.json'), os.path.join(BT, 'availability.csv')] + \
        [os.path.join(BT, f'{p}_{t}{e}') for t in SENS for p, e in (('inputs', '.csv.gz'), ('days', '.jsonl'))] + \
        sorted(glob.glob(os.path.join(DATA, 'era5', 'era5_*.nc'))) + [WEIGHTS]
    envs = {}
    for name, p in ENVS.items():
        r = subprocess.run([os.path.expanduser(p), '-m', 'pip', 'freeze'], capture_output=True, text=True)
        if r.returncode: fail(f'pip freeze in {name} failed')
        envs[name] = r.stdout.split()
    return dict(
        tag=TAG, frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        private_parent_commit=git(PRIVATE, 'rev-parse', 'HEAD')[1], public_parent_commit=git(PUBLIC, 'rev-parse', 'HEAD')[1],
        code_sha256={os.path.relpath(p, PRIVATE): S.sha256(p) for p in code},
        files_sha256={tilde(p): S.sha256(p) for p in files},
        weights_sha256=WEIGHTS_SHA, model_weights_sha256=study['weights_sha256'],
        revisions={k: list(v) for k, v in M.REVISIONS.items()},
        power_json_sha256=S.sha256(os.path.join(PUBLIC, PUBLIC_DIR, 'power.json')),
        cycle_changes_sha256_at_freeze=S.sha256(os.path.join(PUBLIC, PUBLIC_DIR, 'cycle_changes.csv')),
        backtest=dict(days_built=b['days_built'], days_dropped=b['days_dropped'], fallback_days=b['fallback_days']),
        seeds=SEEDS, python=platform.python_version(), environments=envs, tests=tests,
        prospective_window=['2026-11-20', '2027-03-31'])


def main(tag):
    lock = checks()
    show = {k: v for k, v in lock.items() if k not in ('environments', 'files_sha256', 'code_sha256')}
    print(json.dumps(show, indent=1)); print(f'{len(lock["code_sha256"])} code files, {len(lock["files_sha256"])} data files hashed')
    if not tag: print('\nAll checks pass. Run `python weather/freeze_weather.py --tag` to freeze.'); return
    msg = 'Freeze: weather study plan, processing and lock, before any forecast with real weather'
    p = os.path.join(PRIVATE, 'WEATHER_LOCK.json'); json.dump(lock, open(p, 'w'), indent=1)
    for step in (('add', 'WEATHER_LOCK.json'), ('commit', '-q', '-m', msg)):
        c, o = git(PRIVATE, *step)
        if c: sys.exit(f'FREEZE INCOMPLETE at git {" ".join(step)} ({PRIVATE}):\n{o}')
    lock_pub = dict(lock, private_commit=git(PRIVATE, 'rev-parse', 'HEAD')[1])
    json.dump(lock_pub, open(os.path.join(PUBLIC, PUBLIC_DIR, 'WEATHER_LOCK.json'), 'w'), indent=1)
    shutil.copyfile(os.path.join(PRIVATE, 'docs', 'PLAN_weather.md'), os.path.join(PUBLIC, PUBLIC_DIR, 'PLAN.md'))
    steps = [(PUBLIC, ('add', f'{PUBLIC_DIR}/WEATHER_LOCK.json', f'{PUBLIC_DIR}/PLAN.md')), (PUBLIC, ('commit', '-q', '-m', msg))]
    for repo in (PRIVATE, PUBLIC):                             # the daily tasks also push to the public repository
        steps += [(repo, ('pull', '--rebase', '-q')), (repo, ('tag', '-a', TAG, '-m', f'Weather study frozen {lock["frozen_utc"]}')),
                  (repo, ('push', 'origin', 'HEAD')), (repo, ('push', 'origin', TAG))]
    for repo, step in steps:
        c, o = git(repo, *step)
        if c: sys.exit(f'FREEZE INCOMPLETE at git {" ".join(step)} ({repo}):\n{o}\nFix this, then push commits and tags by hand.')
    for repo in (PRIVATE, PUBLIC):
        git(repo, 'fetch', '-q', 'origin', '--tags')
        if git(repo, 'merge-base', '--is-ancestor', 'HEAD', '@{u}')[0] or TAG not in git(repo, 'ls-remote', '--tags', 'origin', TAG)[1]:
            sys.exit(f'FREEZE INCOMPLETE: GitHub does not show {repo}\'s commit and tag yet; push them by hand.')
    print(f'\nFrozen: tag {TAG} in both repositories, confirmed on GitHub. Schedule the daily push next.')


if __name__ == '__main__':
    main('--tag' in sys.argv)
