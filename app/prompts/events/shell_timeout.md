[SYSTEM GUIDANCE — SHELL TIMEOUT]

This shell command exceeded its timeout. Check whether its child processes are
still running and inspect saved output before retrying, so you do not duplicate
work — the timeout kills the process group but a detached grandchild may still
be running.

If the work needs longer than the timeout, launch it in the background from
the start, with output saved to a file and a recorded PID:

    nohup COMMAND > mind/space/JOB-LOG.txt 2>&1 & echo $! > mind/space/JOB.pid

Return promptly and continue other useful work. Check progress later with a
brief `run_shell` call (`tail mind/space/JOB-LOG.txt`, or
`kill -0 $(cat mind/space/JOB.pid)` to confirm it is still alive), or use
`schedule_task` to come back and check the log a few minutes ahead. Do not
block another shell call with a long sleep, wait, or polling loop.
