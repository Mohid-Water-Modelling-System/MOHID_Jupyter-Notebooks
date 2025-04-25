copy ..\data\Nomfich_1.dat Nomfich.dat
mpiexec -np  4 ..\..\..\MOHIDWater_v24.10_x64_MPI\MOHIDWater_v24.10_x64_MPI > display1.txt
..\..\..\MOHIDWater_v24.10_x64_MPI\DomainConsolidation.exe
pause