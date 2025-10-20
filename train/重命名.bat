@echo off
setlocal EnableDelayedExpansion

set a=0

for %%n in (*.mp4) do (
    set "filename=video!a!"
    ren "%%n" "!filename!.mp4"
    set /A a+=1
)

endlocal