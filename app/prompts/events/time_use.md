[SYSTEM NOTIFICATION — TIME USE SINCE THE PREVIOUS CHECKPOINT]

Measured by the harness from the lifetime log:

{{time_use}}

These are facts about how the period was spent, not a verdict. When waiting,
slow commands, or the same command run again and again take most of the time,
shortening that loop (an incremental build, a smaller test case, a cached
intermediate result, work done while something runs) may be worth more than the
next attempt. When the time went into the work itself, no change is needed.
