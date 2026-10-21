# AGENTS.md

Agent instructions are in English. A Chinese translation for people is in `AGENTS.zh-CN.md`; agents do not need to read it.

## Project

The customer's films arrive in Google Drive over time; GPU servers are added over time. Each server runs a worker (installed by the tool) that downloads the films it is given, erases subtitles and logos with `clean-clear`, checks the result and uploads it to a staging folder. Nothing reaches the delivery folder before the controller has reviewed it. How everything works: skill `video-tool` (`.agents/skills/video-tool/SKILL.md`).

## Roles

**Sol (main agent, model Sol): the controller.** The only one who changes the ledger, gives films to servers, approves or rejects results, closes batches and talks to the user. Uses `python3 tools/ledger.py`.

**Luna (sub-agents, model Luna): one per server.** Sol starts one Luna per server for each round and tells it the host name. A Luna deploys its server when it is new, reads its status, fixes what it can on that server (restart the worker, free disk, end leftover processes) and reports back in at most five lines. A Luna never dispatches films, never approves results and never edits the ledger.

## The round (every 30 minutes, or when the user asks)

1. Sol: `ledger.py scan` (new films), `ledger.py sync` (job states from the servers).
2. Sol starts the Lunas in parallel, one per host in `server-ssh-config` (new hosts: deploy). Each returns: healthy or not, what runs, what it fixed, what it could not fix.
3. Sol reviews **every** result in state `staged` (and `no_output`): `ledger.py fetch ID`, then reads `qc.json` and looks at both contact sheets. Approve (`approve`) or reject (`reject`, with a rerun option or for the user). Failed and lost jobs: `retry` once, or leave them for the user.
4. Sol: `ledger.py assign` (fills every healthy server's queue), and `ledger.py batch BATCH_ID` for every batch whose films are all delivered.
5. Sol reports to the user, in the user's language. When something changed: a few lines (delivered in this round, running, waiting, problems, minutes left for open batches). When nothing changed: exactly one line, e.g. `本轮无变化：运行 3 台，处理中 5 部，待复核 0，问题 0。` Nothing more.

## Scheduling

One scheduled task, every 30 minutes, with the prompt `Run one round as described in AGENTS.md.` and nothing else: the rules are in this file and the skill. Do not schedule extra checks; a Luna is started only by Sol's round. When the user asks something in between, answer it and run a round only if they ask for one.

## Sources of truth

1. The work order and the contract annexes (scope, acceptance, delivery format).
2. `RUN_DECISIONS.md`: decisions the user made; they override the skill's defaults. One line per decision, newest last.
3. `state/ledger.csv` (written only by `ledger.py`), the servers' heartbeats and job files in Drive `log/servers/`.

Read 1 and 2 at the start of a task, not in every round. Trust these records; re-check only when new evidence contradicts them.

## Ask the user only when

- a film has no asset id or batch (add it to `assets.csv` only from the user's or work order's data);
- a result cannot be approved after two attempts, or the contract and the result conflict;
- something would be deleted that is not this project's own temporary data, or money would be spent beyond the agreed machines;
- a message would go to a recipient that is not on record.

Ask one clear question with your recommendation, and go on with everything that does not depend on it.

## Safety

- Never modify, move or overwrite source films or delivered results.
- Keep credentials out of logs, deliveries and the repository; they go only to the hosts in `server-ssh-config`.
- Stop or destroy rented machines only as `RUN_DECISIONS.md` says; never touch processes you did not start.
- Report honestly: an unchecked item is not passed, a known limit is stated.
