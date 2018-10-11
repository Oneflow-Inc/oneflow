#!/bin/bash

function GetPID #Name
{
   PsName=$1
   pid=`ps -u $USER | grep $PsName | sed -n 1p | awk '{print $1}'`
   echo $pid
}


function KillProcess #Name
{
  PsName=$1
  PID=`GetPID $PsName`
  while [ "-$PID" != "-" ]
  do
    kill -9 $PID
    echo "Kill $PsName process PID: $PID"
    sleep 1
    PID=`GetPID $PsName`
  done
}

KillProcess oneflow
