# ========================================================
# The tool is used to export image by inputting neural args
# --------------------------------------------------------
# Author: Huailiang.Peng
# Data:   2019.09.19
# =========================================================
#!/bin/sh


UNITY=/Applications/Unity/Unity.app/Contents/MacOS/Unity

cd `dirname $0`

cd ../unity

PROJPATH=`pwd`

cd ../export

EXPPATH=`pwd`

echo ${PROJPATH}

echo ${EXPPATH}

$UNITY -projectPath ${PROJPATH}  -logFile /tmp/env.log -executeMethod XEditor.NeuralInterface.SetupEnv -batchmode

sleep 1

echo ${UNITY}

$UNITY -projectPath ${PROJPATH}  -logFile /tmp/nerual.log -executeMethod XEditor.NeuralInterface.Batch -quit -batchmode

open ${EXPPATH}

