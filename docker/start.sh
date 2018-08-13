#!/bin/bash

# source hadoop-env.sh
source ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

# sed hadoop config file with HADOOP_BACKEND
sed -Ei "s#HADOOP_BACKEND#$HADOOP_BACKEND#g" ${HADOOP_HOME}/etc/hadoop/core-site.xml
sed -Ei "s#HADOOP_BACKEND#$HADOOP_BACKEND#g" ${HADOOP_HOME}/etc/hadoop/hdfs-site.xml

# start yarn & dfs
${HADOOP_HOME}/sbin/start-all.sh > /dev/null 2>&1

# start oneflow train job
/opt/bin/oneflow $*