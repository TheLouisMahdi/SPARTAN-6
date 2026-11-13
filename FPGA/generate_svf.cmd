setMode -bscan
setCable -port svf -file eye_top.svf
addDevice -p 1 -file eye_top.bit
program -p 1
quit
